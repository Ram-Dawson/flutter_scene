import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_scene/src/components/spot_light_component.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/light.dart';
import 'package:flutter_scene/src/render/mesh_draw_selection.dart'
    show hasMeshDrawSelector;
import 'package:flutter_scene/src/render/render_scene.dart' show RenderItem;
import 'package:flutter_scene/src/render/spot_shadow.dart'
    show SpotShadowFrame, kMaxSpotShadows, kMinShadowLightIntensity;
import 'package:vector_math/vector_math.dart';

void _releaseTexture(gpu.Texture? texture) {
  if (texture != null) {
    gpu.gpuHost.releaseTexture(texture);
  }
}

/// Whether [light] actively casts cascaded shadows (`castsShadow` enabled and
/// [DirectionalLight.intensity] above [kMinShadowLightIntensity]).
bool directionalLightCastsShadow(DirectionalLight? light) =>
    light != null &&
    light.castsShadow &&
    light.intensity > kMinShadowLightIntensity;

/// One cascade's cached static-caster shadow tile: the persistent texture the
/// static geometry was rendered into, the light-space matrix it was rendered
/// with, and the coverage it was fit to.
class ShadowCascadeCacheEntry {
  /// The persistent tile texture, allocated lazily by the shadow pass on the
  /// entry's first refresh (kept out of [DirectionalShadowCache.plan] so the
  /// planning logic stays GPU-free).
  gpu.Texture? tile;

  /// World -> light-clip matrix the tile's content was rendered with. Every
  /// consumer (dynamic casters, the lit shader, custom passes) samples through
  /// this matrix, not the frame's ideal one, so the cached content stays
  /// correct while it is reused.
  final Matrix4 matrix = Matrix4.zero();

  /// The ideal bounding-sphere the slack box was built around.
  final Vector3 center = Vector3.zero();
  double radius = 0.0;

  /// Side length of the slack-enlarged orthographic box in world units.
  double boxSize = 0.0;

  /// The static-content signature the tile was rendered with; a mismatch
  /// marks the tile stale (refreshed amortized).
  int renderedSignature = 0;

  /// The normalized light direction the tile was rendered with; a mismatch
  /// marks the tile stale (refreshed amortized).
  final Vector3 direction = Vector3.zero();

  /// Whether the tile has ever been rendered with the current parameters.
  bool hasContent = false;
}

/// A static tile the shadow pass must (re)render this frame.
class ShadowTileRefresh {
  ShadowTileRefresh(this.cascadeIndex, this.entry);

  final int cascadeIndex;
  final ShadowCascadeCacheEntry entry;
}

/// One frame's cached-shadow decisions: the cascades every consumer samples
/// with (cached matrices, this frame's split distances) and the tiles to
/// re-render.
class ShadowCachePlan {
  ShadowCachePlan(this.cascades, this.refreshes, this.entries, this.cache);

  /// Effective cascades. Matrices and box sizes describe the cached tiles;
  /// split distances are the frame's ideal ones (they only select which
  /// cascade a fragment samples).
  final List<ShadowCascade> cascades;

  /// Tiles to render static casters into this frame, in cascade order.
  final List<ShadowTileRefresh> refreshes;

  /// All cache entries, indexed by cascade.
  final List<ShadowCascadeCacheEntry> entries;

  /// The cache that made this plan (owns the shared tile depth).
  final DirectionalShadowCache cache;
}

/// Cross-frame cache for the directional light's cascaded shadow tiles.
///
/// Static casters ([RenderItem.shadowStatic]) render into persistent per
/// cascade tiles that are reused while they still cover the view; the shadow
/// pass composites them into each frame's atlas and draws only the dynamic
/// casters on top. Tiles are fit with [slackFactor] extra radius so the
/// camera can move and turn inside the slack before a cascade must
/// re-render. Stale tiles (a static caster appeared or vanished, or the light
/// turned by up to [maxDirectionLagDegrees]) refresh at most
/// [maxAmortizedRefreshes] per frame, nearest cascade first, so streaming
/// worlds and a stepped sun never pay for every cascade at once. A stale tile
/// keeps sampling through the matrix it was rendered with until it refreshes.
class DirectionalShadowCache {
  /// How much larger than the ideal bounding sphere each tile is rendered.
  /// Costs ~13% effective resolution; buys re-render-free camera movement
  /// within the slack.
  static const double slackFactor = 1.15;

  /// Upper bound on stale-but-usable tile refreshes per frame.
  static const int maxAmortizedRefreshes = 1;

  /// How far the light may turn from a tile's direction before that tile
  /// re-renders immediately instead of amortized.
  static const double maxDirectionLagDegrees = 5.0;

  static final double _minDirectionLagCos = math.cos(
    maxDirectionLagDegrees * math.pi / 180.0,
  );

  final List<ShadowCascadeCacheEntry> _entries = [];
  int _resolution = 0;
  ShadowCasterFaces _casterFaces = ShadowCasterFaces.front;
  int _casterChannelMask = 0xFF;

  /// The depth attachment every tile refresh renders with, allocated lazily by
  /// the shadow pass and dropped with the tiles on a resolution change.
  ///
  /// Backends cache a framebuffer per color texture (flutter/flutter#192538),
  /// so a tile must keep the depth it was first rendered with for as long as
  /// it lives. A pooled depth rotates and is freed on resize or memory
  /// pressure, leaving the cached framebuffer attached to a released texture.
  gpu.Texture? tileDepth;

  /// Releases all persistent tile textures and resets cached state.
  void dispose() {
    for (final entry in _entries) {
      _releaseTexture(entry.tile);
      entry.tile = null;
      entry.hasContent = false;
    }
    _entries.clear();
    _releaseTexture(tileDepth);
    tileDepth = null;
    _resolution = 0;
  }

  /// Decides which tiles to re-render for this frame's [idealCascades] and
  /// returns the effective cascades to sample with.
  ///
  /// [staticSignature] fingerprints the static caster set; any change marks
  /// every tile stale, as does a small turn of [lightDirection]. A larger turn
  /// or a change to the shadow parameters re-renders every tile this frame,
  /// keeping the tile textures unless the resolution changed.
  ShadowCachePlan plan({
    required DirectionalLight light,
    required Vector3 lightDirection,
    required List<ShadowCascade> idealCascades,
    required int staticSignature,
  }) {
    final resolution = light.shadowMapResolution;
    final dir = lightDirection.normalized();
    final paramsChanged =
        resolution != _resolution ||
        light.shadowCasterFaces != _casterFaces ||
        light.shadowCasterChannelMask != _casterChannelMask ||
        _entries.length != idealCascades.length;
    if (paramsChanged) {
      if (resolution != _resolution) {
        for (final entry in _entries) {
          _releaseTexture(entry.tile);
          entry.tile = null;
        }
        _releaseTexture(tileDepth);
        tileDepth = null;
      }
      // Kept entries keep their tile textures, which still pair with
      // [tileDepth].
      if (_entries.length > idealCascades.length) {
        for (var i = idealCascades.length; i < _entries.length; i++) {
          _releaseTexture(_entries[i].tile);
          _entries[i].tile = null;
        }
        _entries.length = idealCascades.length;
      }
      for (final entry in _entries) {
        entry.hasContent = false;
      }
      while (_entries.length < idealCascades.length) {
        _entries.add(ShadowCascadeCacheEntry());
      }
      _resolution = resolution;
      _casterFaces = light.shadowCasterFaces;
      _casterChannelMask = light.shadowCasterChannelMask;
    }

    final refreshes = <ShadowTileRefresh>[];
    final effective = <ShadowCascade>[];
    var amortized = 0;
    for (var i = 0; i < idealCascades.length; i++) {
      final ideal = idealCascades[i];
      final entry = _entries[i];
      final center = ideal.center ?? Vector3.zero();
      // A tile is reusable while the ideal sphere still fits inside its
      // slack box; the radius only changes with camera/shadow parameters.
      final directionCos = entry.direction.dot(dir);
      final fits =
          entry.hasContent &&
          directionCos >= _minDirectionLagCos &&
          (ideal.radius - entry.radius).abs() <= entry.radius * 1e-3 &&
          (center - entry.center).length <= entry.radius * (slackFactor - 1.0);
      final stale =
          entry.renderedSignature != staticSignature ||
          entry.direction.distanceToSquared(dir) > 1e-10;
      var refresh = false;
      if (!fits) {
        // Unusable (first render, coverage drift, a large turn, or a
        // parameter change): must render this frame or the cascade has no
        // shadows.
        refresh = true;
      } else if (stale && amortized < maxAmortizedRefreshes) {
        // Usable but stale: refresh a bounded number per frame,
        // nearest cascade first (this loop runs near-to-far).
        refresh = true;
        amortized++;
      }
      if (refresh) {
        entry.center.setFrom(center);
        entry.radius = ideal.radius;
        entry.boxSize = ideal.radius * slackFactor * 2.0;
        entry.matrix.setFrom(
          light.cascadeLightSpaceMatrix(
            dir,
            center,
            ideal.radius * slackFactor,
          ),
        );
        entry.renderedSignature = staticSignature;
        entry.direction.setFrom(dir);
        entry.hasContent = true;
        refreshes.add(ShadowTileRefresh(i, entry));
      }
      effective.add(
        ShadowCascade(
          lightSpaceMatrix: entry.matrix,
          splitDistance: ideal.splitDistance,
          boxSize: entry.boxSize,
          center: entry.center,
          radius: entry.radius,
        ),
      );
    }
    return ShadowCachePlan(effective, refreshes, _entries, this);
  }
}

/// Which persistent texture supplies a cached spot shadow slot's atlas tile.
enum SpotTileCopySource {
  /// Neither static nor dynamic casters intersect this spot's channels; the
  /// cleared atlas tile (depth `1.0`) is already valid.
  none,

  /// [SpotShadowCacheEntry.staticTile] holds the complete shadow map (static
  /// casters present, zero dynamic casters).
  staticTile,

  /// [SpotShadowCacheEntry.completeTile] holds the composite of static and
  /// dynamic casters (or dynamic casters alone).
  completeTile,
}

/// Cached shadow tiles and invalidation state for one [SpotLightComponent].
class SpotShadowCacheEntry {
  /// Persistent static-caster shadow tile (`ShadowCasterFilter.staticOnly`),
  /// allocated lazily by the shadow pass so planning remains GPU-free.
  gpu.Texture? staticTile;

  /// Persistent complete shadow tile (`staticTile` + `dynamicOnly`, or
  /// `dynamicOnly` when no static casters exist), reused directly while dynamic
  /// casters remain stationary.
  gpu.Texture? completeTile;

  /// World -> spot-clip matrix the cached tile content was rendered with.
  final Matrix4 matrix = Matrix4.zero();

  /// Whether [matrix] has been initialized from a frame.
  bool hasMatrix = false;

  /// Tile edge resolution in texels when the cached content was rendered.
  int tileResolution = 0;

  /// Caster-face mode when the cached content was rendered.
  ShadowCasterFaces casterFaces = ShadowCasterFaces.front;

  /// Shadow-caster channel mask when the cached content was rendered.
  int casterChannelMask = 0xFF;

  /// Static-caster signature rendered into [staticTile].
  int renderedStaticSignature = 0;

  /// Dynamic-caster signature rendered into [completeTile].
  int lastDynamicSignature = 0;

  /// Whether static casters were composited into [completeTile].
  bool lastHasStaticCasters = false;

  /// Whether [staticTile] holds valid content for the current parameters.
  bool hasStaticContent = false;

  /// Whether [completeTile] holds valid content for the current parameters.
  bool hasCompleteContent = false;

  /// Releases only [staticTile] and resets [hasStaticContent].
  void releaseStaticTile() {
    _releaseTexture(staticTile);
    staticTile = null;
    hasStaticContent = false;
  }

  /// Releases only [completeTile] and resets [hasCompleteContent].
  void releaseCompleteTile() {
    _releaseTexture(completeTile);
    completeTile = null;
    hasCompleteContent = false;
  }

  /// Releases all owned GPU textures and marks cached content invalid.
  void releaseTextures() {
    releaseStaticTile();
    releaseCompleteTile();
  }
}

/// Per-slot execution plan for one spot shadow caster in [SpotShadowCachePlan].
class SpotShadowTilePlan {
  SpotShadowTilePlan({
    required this.slot,
    required this.component,
    required this.matrix,
    required this.casterFaces,
    required this.casterChannelMask,
    required this.cached,
    required this.entry,
    required this.hasStaticCasters,
    required this.hasDynamicCasters,
    required this.renderStaticTile,
    required this.renderCompleteTile,
    required this.copySource,
  });

  /// Creates an uncached slot plan that renders all casters directly into the
  /// frame's shadow atlas.
  SpotShadowTilePlan.uncached({
    required int slot,
    required SpotLightComponent component,
    required Matrix4 matrix,
    required ShadowCasterFaces casterFaces,
    required int casterChannelMask,
  }) : this(
         slot: slot,
         component: component,
         matrix: matrix,
         casterFaces: casterFaces,
         casterChannelMask: casterChannelMask,
         cached: false,
         entry: null,
         hasStaticCasters: false,
         hasDynamicCasters: false,
         renderStaticTile: false,
         renderCompleteTile: false,
         copySource: SpotTileCopySource.none,
       );

  /// Slot index within [SpotShadowFrame.casters] (`0 ..< casters.length`).
  final int slot;

  /// The spot light component owning this slot.
  final SpotLightComponent component;

  /// Effective world -> spot-clip matrix for this slot.
  final Matrix4 matrix;

  /// Which faces cast into this slot's shadow map.
  final ShadowCasterFaces casterFaces;

  /// Channel mask selecting which nodes cast into this slot's shadow map.
  final int casterChannelMask;

  /// Whether this slot participates in [SpotShadowCache] (`false` when
  /// [SpotLight.cacheStaticShadows] is disabled).
  final bool cached;

  /// Persistent cache entry for [component], or `null` when [cached] is false.
  final SpotShadowCacheEntry? entry;

  /// Whether static shadow casters are present in the scene this frame.
  final bool hasStaticCasters;

  /// Whether any visible dynamic shadow caster intersects [casterChannelMask].
  final bool hasDynamicCasters;

  /// Whether the shadow pass must render `staticOnly` casters into
  /// [SpotShadowCacheEntry.staticTile] this frame.
  bool renderStaticTile;

  /// Whether the shadow pass must rebuild [SpotShadowCacheEntry.completeTile]
  /// this frame (`staticTile` copy when [hasStaticCasters], plus `dynamicOnly`
  /// casters on top).
  bool renderCompleteTile;

  /// Which cached tile texture to blit into the frame's shared shadow atlas.
  final SpotTileCopySource copySource;
}

/// One frame's cached spot-shadow execution plan across active spot shadow
/// slots.
class SpotShadowCachePlan {
  SpotShadowCachePlan({
    required this.tiles,
    required this.tileResolution,
    required this.cache,
  });

  /// Per-slot plans parallel to [SpotShadowFrame.casters].
  final List<SpotShadowTilePlan> tiles;

  /// Tile resolution in texels shared by all tiles in [cache].
  final int tileResolution;

  /// The owning [SpotShadowCache] (holds the shared [SpotShadowCache.tileDepth]
  /// attachment).
  final SpotShadowCache cache;

  /// Number of slots refreshing their [SpotShadowCacheEntry.staticTile].
  int get staticRefreshCount {
    var count = 0;
    for (final tile in tiles) {
      if (tile.renderStaticTile) count++;
    }
    return count;
  }

  /// Number of slots refreshing their [SpotShadowCacheEntry.completeTile].
  int get completeRefreshCount {
    var count = 0;
    for (final tile in tiles) {
      if (tile.renderCompleteTile) count++;
    }
    return count;
  }

  /// Whether any slot needs a static or complete tile render pass this frame.
  bool get hasRefreshes => staticRefreshCount > 0 || completeRefreshCount > 0;
}

int _floatHash(double value) => (value == 0.0 ? 0.0 : value).hashCode;

int _hashDynamicShadowItem(RenderItem item) {
  var h = identityHashCode(item.geometry);
  h = 0x3fffffff & (h * 31 + identityHashCode(item.material));
  final instances = item.instanceTransforms;
  h = 0x3fffffff & (h * 31 + identityHashCode(instances));
  if (instances != null) {
    h = 0x3fffffff & (h * 31 + instances.length);
    final bounds = item.worldBounds;
    if (bounds != null) {
      final min = bounds.min;
      final max = bounds.max;
      h = 0x3fffffff & (h * 31 + _floatHash(min.x));
      h = 0x3fffffff & (h * 31 + _floatHash(min.y));
      h = 0x3fffffff & (h * 31 + _floatHash(min.z));
      h = 0x3fffffff & (h * 31 + _floatHash(max.x));
      h = 0x3fffffff & (h * 31 + _floatHash(max.y));
      h = 0x3fffffff & (h * 31 + _floatHash(max.z));
    }
  }
  h = 0x3fffffff & (h * 31 + identityHashCode(item.jointsTexture));
  final morphWeights = item.morphWeights;
  if (morphWeights != null) {
    h = 0x3fffffff & (h * 31 + morphWeights.length);
    for (var i = 0; i < morphWeights.length; i++) {
      h = 0x3fffffff & (h * 31 + _floatHash(morphWeights[i]));
    }
  }
  h = 0x3fffffff & (h * 31 + item.shadowCastingMode.index);
  h = 0x3fffffff & (h * 31 + (item.nodeWindingFlipped ? 1 : 0));
  final storage = item.worldTransform.storage;
  for (var i = 0; i < 16; i++) {
    h = 0x3fffffff & (h * 31 + _floatHash(storage[i]));
  }
  return h;
}

/// Computes the dynamic shadow-caster signature across [items] for a spot light
/// with [casterChannelMask].
({bool hasDynamicCasters, bool hasUncacheableDynamicCasters, int signature})
computeSpotDynamicSignature(List<RenderItem> items, int casterChannelMask) {
  if (casterChannelMask == 0) {
    return (
      hasDynamicCasters: false,
      hasUncacheableDynamicCasters: false,
      signature: 0,
    );
  }
  var hasDynamic = false;
  var hasUncacheable = false;
  var sig = 0;
  for (var i = 0; i < items.length; i++) {
    final item = items[i];
    if (item.shadowStatic || !item.castsShadows || !item.visible) continue;
    if ((item.lightChannelMask & casterChannelMask) == 0) continue;
    hasDynamic = true;
    if (hasMeshDrawSelector(item)) {
      hasUncacheable = true;
    }
    sig = 0x3fffffff & (sig * 37 + _hashDynamicShadowItem(item));
  }
  return (
    hasDynamicCasters: hasDynamic,
    hasUncacheableDynamicCasters: hasUncacheable,
    signature: sig,
  );
}

bool _matricesClose(Matrix4 a, Matrix4 b, double tolerance) {
  final sa = a.storage;
  final sb = b.storage;
  for (var i = 0; i < 16; i++) {
    final diff = (sa[i] - sb[i]).abs();
    if (diff.isNaN || diff > tolerance) return false;
  }
  return true;
}

/// Cross-frame two-level shadow tile cache for shadow-casting [SpotLight]s.
///
/// Keyed by [SpotLightComponent] identity so slot reordering never thrashes
/// cached tiles, with automatic pruning when a spot leaves the active caster
/// set or disables [SpotLight.cacheStaticShadows]:
///
/// 1. **Static caster tile ([SpotShadowCacheEntry.staticTile])**: caches
///    `shadowStatic` casters while the spot's clip matrix (within
///    [matrixTolerance]), tile resolution, caster faces, channel mask, and
///    static caster signature remain unchanged. When no dynamic casters match
///    the spot's channel mask, [SpotShadowCacheEntry.staticTile] IS the
///    complete shadow map and blits directly into the frame atlas with zero BVH
///    queries and zero caster draw calls.
/// 2. **Stationary full-tile reuse ([SpotShadowCacheEntry.completeTile])**:
///    when dynamic casters exist, composites `staticTile` + `dynamicOnly`
///    casters into `completeTile` and reuses `completeTile` directly via a tile
///    copy on subsequent frames whenever the dynamic caster signature holds
///    steady (e.g. during camera orbit with stationary props).
class SpotShadowCache {
  /// Element-wise tolerance on the 16 elements of [Matrix4.storage] when
  /// testing whether a spot's shadow clip matrix is unchanged.
  static const double matrixTolerance = 1e-6;

  final Map<SpotLightComponent, SpotShadowCacheEntry> _entries = Map.identity();
  int _resolution = 0;

  Int32List _slotDynamicSigs = Int32List(kMaxSpotShadows);
  Uint8List _slotHasDynamic = Uint8List(kMaxSpotShadows);
  Uint8List _slotUncacheableDynamic = Uint8List(kMaxSpotShadows);

  /// Shared transient depth attachment paired with every tile texture in this
  /// cache, matching [DirectionalShadowCache.tileDepth] so backend framebuffer
  /// caches (flutter/flutter#192538) never outlive their depth attachment.
  gpu.Texture? tileDepth;

  /// Number of active component entries currently retained by the cache.
  int get entryCount => _entries.length;

  /// Returns the cached entry for [component], or `null` if none is retained.
  SpotShadowCacheEntry? entryFor(SpotLightComponent component) =>
      _entries[component];

  /// Whether this cache currently holds any allocated GPU textures.
  bool get hasTextures {
    if (tileDepth != null) return true;
    for (final entry in _entries.values) {
      if (entry.staticTile != null || entry.completeTile != null) {
        return true;
      }
    }
    return false;
  }

  /// Releases all owned GPU textures (`staticTile`, `completeTile`, and
  /// [tileDepth]) and clears all cached entries.
  void dispose() {
    for (final entry in _entries.values) {
      entry.releaseTextures();
    }
    _entries.clear();
    _releaseTexture(tileDepth);
    tileDepth = null;
    _resolution = 0;
  }

  /// Plans cached tile refreshes and atlas copy sources for [spotShadows].
  ///
  /// [tileResolution] overrides [SpotShadowFrame.tileResolution] when the
  /// shared shadow atlas uses a directional light's resolution. Planning stays
  /// GPU-free; [ShadowPass] lazily allocates textures when executing the plan.
  SpotShadowCachePlan plan({
    required SpotShadowFrame spotShadows,
    required List<RenderItem> items,
    required bool hasStaticShadowCasters,
    required int staticSignature,
    int? tileResolution,
  }) {
    final effectiveResolution = tileResolution ?? spotShadows.tileResolution;
    if (_resolution != effectiveResolution) {
      for (final entry in _entries.values) {
        entry.releaseTextures();
      }
      _releaseTexture(tileDepth);
      tileDepth = null;
      _resolution = effectiveResolution;
    }

    final casters = spotShadows.casters;
    final slotCount = casters.length;

    // Prune components no longer in the active caster list or that opted out
    // of static shadow caching.
    if (_entries.isNotEmpty) {
      List<SpotLightComponent>? staleKeys;
      for (final key in _entries.keys) {
        var keep = false;
        for (var i = 0; i < slotCount; i++) {
          if (identical(casters[i], key) && key.light.cacheStaticShadows) {
            keep = true;
            break;
          }
        }
        if (!keep) {
          (staleKeys ??= []).add(key);
        }
      }
      if (staleKeys != null) {
        for (final key in staleKeys) {
          _entries.remove(key)?.releaseTextures();
        }
      }
    }

    if (slotCount > _slotDynamicSigs.length) {
      _slotDynamicSigs = Int32List(slotCount);
      _slotHasDynamic = Uint8List(slotCount);
      _slotUncacheableDynamic = Uint8List(slotCount);
    }

    var unionChannelMask = 0;
    for (var s = 0; s < slotCount; s++) {
      _slotDynamicSigs[s] = 0;
      _slotHasDynamic[s] = 0;
      _slotUncacheableDynamic[s] = 0;
      if (casters[s].light.cacheStaticShadows) {
        unionChannelMask |= spotShadows.casterChannelMasks[s];
      }
    }

    // Single pass over render items to compute per-slot dynamic caster
    // signatures and presence flags for all cached spots at once.
    if (unionChannelMask != 0) {
      for (var i = 0; i < items.length; i++) {
        final item = items[i];
        if (item.shadowStatic || !item.castsShadows || !item.visible) continue;
        final itemChannels = item.lightChannelMask;
        if ((itemChannels & unionChannelMask) == 0) continue;

        final itemHash = _hashDynamicShadowItem(item);
        final uncacheable = hasMeshDrawSelector(item);
        for (var s = 0; s < slotCount; s++) {
          if (!casters[s].light.cacheStaticShadows) continue;
          if ((itemChannels & spotShadows.casterChannelMasks[s]) == 0) continue;
          _slotHasDynamic[s] = 1;
          if (uncacheable) _slotUncacheableDynamic[s] = 1;
          _slotDynamicSigs[s] =
              0x3fffffff & (_slotDynamicSigs[s] * 37 + itemHash);
        }
      }
    }

    final tilePlans = <SpotShadowTilePlan>[];
    for (var s = 0; s < slotCount; s++) {
      final component = casters[s];
      final light = component.light;
      final matrix = spotShadows.matrices[s];
      final casterFaces = spotShadows.casterFacesFor(s);
      final channelMask = spotShadows.casterChannelMasks[s];

      if (!light.cacheStaticShadows) {
        tilePlans.add(
          SpotShadowTilePlan.uncached(
            slot: s,
            component: component,
            matrix: matrix,
            casterFaces: casterFaces,
            casterChannelMask: channelMask,
          ),
        );
        continue;
      }

      final entry = _entries.putIfAbsent(component, SpotShadowCacheEntry.new);
      final hasStaticForSpot = hasStaticShadowCasters && channelMask != 0;
      final hasDynamic = _slotHasDynamic[s] != 0;
      final dynamicSignature = _slotDynamicSigs[s];
      final uncacheableDynamic = _slotUncacheableDynamic[s] != 0;

      final spotParamsMatch =
          entry.hasMatrix &&
          entry.tileResolution == effectiveResolution &&
          entry.casterFaces == casterFaces &&
          entry.casterChannelMask == channelMask &&
          _matricesClose(entry.matrix, matrix, matrixTolerance);

      var renderStaticTile = false;
      if (hasStaticForSpot) {
        final staticValid =
            spotParamsMatch &&
            entry.hasStaticContent &&
            entry.renderedStaticSignature == staticSignature;
        if (!staticValid) {
          renderStaticTile = true;
          entry.renderedStaticSignature = staticSignature;
          entry.hasStaticContent = true;
        }
      } else if (entry.staticTile != null || entry.hasStaticContent) {
        entry.releaseStaticTile();
      }

      var renderCompleteTile = false;
      final SpotTileCopySource copySource;
      if (hasStaticForSpot && !hasDynamic) {
        // Case 1: only static casters affect this spot, so staticTile IS the
        // complete shadow map.
        if (entry.completeTile != null || entry.hasCompleteContent) {
          entry.releaseCompleteTile();
        }
        copySource = SpotTileCopySource.staticTile;
      } else if (hasDynamic) {
        // Case 2: dynamic casters exist; reuse completeTile across frames
        // while dynamic casters and staticTile remain unchanged.
        final completeValid =
            spotParamsMatch &&
            !renderStaticTile &&
            !uncacheableDynamic &&
            entry.hasCompleteContent &&
            entry.lastHasStaticCasters == hasStaticForSpot &&
            entry.lastDynamicSignature == dynamicSignature;
        if (!completeValid) {
          renderCompleteTile = true;
          entry.lastDynamicSignature = dynamicSignature;
          entry.lastHasStaticCasters = hasStaticForSpot;
          entry.hasCompleteContent = true;
        }
        copySource = SpotTileCopySource.completeTile;
      } else {
        // Neither static nor dynamic casters can cast into this spot; the
        // cleared atlas tile (depth 1.0) is already the complete result.
        if (entry.completeTile != null || entry.hasCompleteContent) {
          entry.releaseCompleteTile();
        }
        copySource = SpotTileCopySource.none;
      }

      if (!spotParamsMatch) {
        entry.matrix.setFrom(matrix);
        entry.hasMatrix = true;
        entry.tileResolution = effectiveResolution;
        entry.casterFaces = casterFaces;
        entry.casterChannelMask = channelMask;
      }

      tilePlans.add(
        SpotShadowTilePlan(
          slot: s,
          component: component,
          matrix: entry.matrix,
          casterFaces: casterFaces,
          casterChannelMask: channelMask,
          cached: true,
          entry: entry,
          hasStaticCasters: hasStaticForSpot,
          hasDynamicCasters: hasDynamic,
          renderStaticTile: renderStaticTile,
          renderCompleteTile: renderCompleteTile,
          copySource: copySource,
        ),
      );
    }

    return SpotShadowCachePlan(
      tiles: tilePlans,
      tileResolution: effectiveResolution,
      cache: this,
    );
  }
}
