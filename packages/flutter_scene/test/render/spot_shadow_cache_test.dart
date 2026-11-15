import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/mesh_draw.dart';
import 'package:flutter_scene/src/render/frame_transients.dart';
import 'package:flutter_scene/src/render/render_graph.dart';
import 'package:flutter_scene/src/render/render_scene.dart';
import 'package:flutter_scene/src/render/shadow_cache.dart';
import 'package:flutter_scene/src/render/shadow_pass.dart';
import 'package:flutter_scene/src/render/spot_shadow.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

import '../support/gpu_available.dart';

class _StubGeometry extends Geometry {
  @override
  void bind(
    gpu.RenderPass pass,
    TransientWriter transientsBuffer,
    Matrix4 modelTransform,
    Matrix4 cameraTransform,
    Vector3 cameraPosition, {
    gpu.Shader? shaderOverride,
    double depthBias = 0.0,
  }) => throw UnsupportedError('stub');
}

class _StubMaterial extends Material {
  @override
  void bind(
    gpu.RenderPass pass,
    TransientWriter transientsBuffer,
    Lighting lighting,
  ) => throw UnsupportedError('stub');
}

class _StubMeshDrawSource implements MeshDrawSource {
  _StubMeshDrawSource(this.drawSelector);

  @override
  final MeshDrawSelector? drawSelector;
}

SpotLightComponent _spot({
  bool castsShadow = true,
  bool cacheStaticShadows = true,
  double intensity = 1.0,
  int resolution = 512,
  ShadowCasterFaces casterFaces = ShadowCasterFaces.front,
  int casterChannelMask = 0xFF,
  Vector3? position,
  Vector3? direction,
}) {
  final node = Node(
    localTransform: Matrix4.translation(position ?? Vector3(0, 5, 0)),
  );
  final component = SpotLightComponent(
    SpotLight(
      castsShadow: castsShadow,
      cacheStaticShadows: cacheStaticShadows,
      intensity: intensity,
      range: 20.0,
      shadowMapResolution: resolution,
      shadowCasterFaces: casterFaces,
      shadowCasterChannelMask: casterChannelMask,
      direction: direction ?? Vector3(0, -1, 0),
    ),
  );
  node.addComponent(component);
  return component;
}

RenderItem _dynamicItem({
  Vector3? position,
  int channelMask = 0xFF,
  bool visible = true,
  ShadowCastingMode shadowCastingMode = ShadowCastingMode.on,
  bool shadowStatic = false,
}) {
  final pos = position ?? Vector3.zero();
  return RenderItem(geometry: _StubGeometry(), material: _StubMaterial())
    ..visible = visible
    ..shadowStatic = shadowStatic
    ..shadowCastingMode = shadowCastingMode
    ..lightChannelMask = channelMask
    ..worldTransform.setTranslation(pos)
    ..worldBounds = Aabb3.minMax(
      pos - Vector3.all(0.5),
      pos + Vector3.all(0.5),
    );
}

void main() {
  group('zero-intensity shadow short-circuit', () {
    test('collectSpotShadows skips spots with intensity <= 1e-4', () {
      final zero = _spot(intensity: 0.0);
      final negative = _spot(intensity: -1.0);
      final atThreshold = _spot(intensity: kMinShadowLightIntensity);
      expect(collectSpotShadows([zero, negative, atThreshold]), isNull);

      final aboveThreshold = _spot(intensity: 1.01e-4);
      final frame = collectSpotShadows([zero, atThreshold, aboveThreshold])!;
      expect(frame.casters, [aboveThreshold]);
      expect(frame.slotOf(zero), -1);
      expect(frame.slotOf(aboveThreshold), 0);
    });

    test('zero-intensity spots do not consume kMaxSpotShadows budget', () {
      final inactive = [for (var i = 0; i < 3; i++) _spot(intensity: 0.0)];
      final active = [
        for (var i = 0; i < kMaxSpotShadows; i++)
          _spot(intensity: 2.0, position: Vector3(i.toDouble(), 5, 0)),
      ];
      final frame = collectSpotShadows([...inactive, ...active])!;
      expect(frame.casters, active);
    });

    test('spotLightCastsShadow and directionalLightCastsShadow helpers', () {
      expect(
        spotLightCastsShadow(SpotLight(castsShadow: false, intensity: 5.0)),
        isFalse,
      );
      expect(
        spotLightCastsShadow(SpotLight(castsShadow: true, intensity: 0.0)),
        isFalse,
      );
      expect(
        spotLightCastsShadow(SpotLight(castsShadow: true, intensity: 1e-4)),
        isFalse,
      );
      expect(
        spotLightCastsShadow(SpotLight(castsShadow: true, intensity: 2e-4)),
        isTrue,
      );

      expect(directionalLightCastsShadow(null), isFalse);
      expect(
        directionalLightCastsShadow(
          DirectionalLight(castsShadow: false, intensity: 3.0),
        ),
        isFalse,
      );
      expect(
        directionalLightCastsShadow(
          DirectionalLight(castsShadow: true, intensity: 0.0),
        ),
        isFalse,
      );
      expect(
        directionalLightCastsShadow(
          DirectionalLight(castsShadow: true, intensity: 1e-4),
        ),
        isFalse,
      );
      expect(
        directionalLightCastsShadow(
          DirectionalLight(castsShadow: true, intensity: 2e-4),
        ),
        isTrue,
      );
    });
  });

  group('SpotLight.cacheStaticShadows', () {
    test('defaults to true and can be configured or mutated', () {
      final defaultSpot = SpotLight();
      expect(defaultSpot.cacheStaticShadows, isTrue);

      final uncachedSpot = SpotLight(cacheStaticShadows: false);
      expect(uncachedSpot.cacheStaticShadows, isFalse);
      uncachedSpot.cacheStaticShadows = true;
      expect(uncachedSpot.cacheStaticShadows, isTrue);
    });
  });

  group('SpotShadowCache planning', () {
    late SpotShadowCache cache;

    setUp(() {
      cache = SpotShadowCache();
    });

    test(
      'Case 1 (static casters only): renders staticTile once and reuses it as complete map',
      () {
        final spot = _spot();
        final staticWall = _dynamicItem(shadowStatic: true);
        final items = [staticWall];

        final f1 = collectSpotShadows([spot])!;
        final p1 = cache.plan(
          spotShadows: f1,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 101,
        );
        expect(p1.tiles, hasLength(1));
        expect(p1.tiles.single.cached, isTrue);
        expect(p1.tiles.single.hasStaticCasters, isTrue);
        expect(p1.tiles.single.hasDynamicCasters, isFalse);
        expect(p1.tiles.single.renderStaticTile, isTrue);
        expect(p1.tiles.single.renderCompleteTile, isFalse);
        expect(p1.tiles.single.copySource, SpotTileCopySource.staticTile);

        // Frame 2+: zero refreshes, blits staticTile directly.
        final f2 = collectSpotShadows([spot])!;
        final p2 = cache.plan(
          spotShadows: f2,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 101,
        );
        expect(p2.hasRefreshes, isFalse);
        expect(p2.tiles.single.renderStaticTile, isFalse);
        expect(p2.tiles.single.renderCompleteTile, isFalse);
        expect(p2.tiles.single.copySource, SpotTileCopySource.staticTile);

        // Static content change refreshes staticTile.
        final p3 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 202,
        );
        expect(p3.tiles.single.renderStaticTile, isTrue);
        expect(p3.tiles.single.renderCompleteTile, isFalse);
        expect(p3.tiles.single.copySource, SpotTileCopySource.staticTile);
      },
    );

    test(
      'Case 2 (static + dynamic casters): reuses completeTile while stationary and refreshes only completeTile on dynamic motion',
      () {
        final spot = _spot();
        final staticRoom = _dynamicItem(shadowStatic: true);
        final dice = _dynamicItem(position: Vector3(0.5, 1.0, -0.2));
        final items = [staticRoom, dice];

        // Frame 1: renders both staticTile and completeTile.
        final p1 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 7,
        );
        expect(p1.tiles.single.renderStaticTile, isTrue);
        expect(p1.tiles.single.renderCompleteTile, isTrue);
        expect(p1.tiles.single.copySource, SpotTileCopySource.completeTile);

        // Frame 2 (camera orbits, props stationary): zero refreshes!
        final p2 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 7,
        );
        expect(p2.hasRefreshes, isFalse);
        expect(p2.tiles.single.renderStaticTile, isFalse);
        expect(p2.tiles.single.renderCompleteTile, isFalse);
        expect(p2.tiles.single.copySource, SpotTileCopySource.completeTile);

        // Frame 3 (dice translates): staticTile stays cached, completeTile refreshes!
        dice.worldTransform.setTranslation(Vector3(0.8, 1.0, -0.2));
        final p3 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 7,
        );
        expect(p3.tiles.single.renderStaticTile, isFalse);
        expect(p3.tiles.single.renderCompleteTile, isTrue);
        expect(p3.tiles.single.copySource, SpotTileCopySource.completeTile);

        // Frame 4 (dice settles at new spot): zero refreshes again!
        final p4 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 7,
        );
        expect(p4.hasRefreshes, isFalse);

        // Frame 5 (dice rotates in place without translation change):
        // upper-left 3x3 of worldTransform.storage changes -> invalidates completeTile!
        dice.worldTransform.setRotation(Matrix3.rotationY(0.35));
        final p5 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 7,
        );
        expect(p5.tiles.single.renderStaticTile, isFalse);
        expect(p5.tiles.single.renderCompleteTile, isTrue);
      },
    );

    test(
      'dynamic caster mutations (morphWeights, material, instances, visibility) invalidate completeTile',
      () {
        final spot = _spot();
        final prop = _dynamicItem();
        final items = [prop];

        SpotShadowCachePlan runPlan() => cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 42,
        );

        expect(runPlan().tiles.single.renderCompleteTile, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        // Attach morph weights and mutate an element in-place.
        final weights = Float32List.fromList([0.0, 0.5]);
        prop.morphWeights = weights;
        expect(runPlan().tiles.single.renderCompleteTile, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        weights[1] = 0.75;
        final afterMorph = runPlan();
        expect(afterMorph.tiles.single.renderStaticTile, isFalse);
        expect(afterMorph.tiles.single.renderCompleteTile, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        // Swap material (e.g. alpha-masked cutout material change).
        prop.material = _StubMaterial();
        expect(runPlan().tiles.single.renderCompleteTile, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        // Assign instance transforms.
        prop.instanceTransforms = [Matrix4.identity(), Matrix4.identity()];
        expect(runPlan().tiles.single.renderCompleteTile, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        // Hide the dynamic caster: falls back to staticTile with zero refreshes!
        prop.visible = false;
        final afterHide = runPlan();
        expect(afterHide.tiles.single.renderStaticTile, isFalse);
        expect(afterHide.tiles.single.renderCompleteTile, isFalse);
        expect(
          afterHide.tiles.single.copySource,
          SpotTileCopySource.staticTile,
        );
      },
    );

    test(
      'dynamic-only scene (no static casters) still caches completeTile while stationary',
      () {
        final spot = _spot();
        final prop = _dynamicItem();

        final p1 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [prop],
          hasStaticShadowCasters: false,
          staticSignature: 0,
        );
        expect(p1.tiles.single.hasStaticCasters, isFalse);
        expect(p1.tiles.single.hasDynamicCasters, isTrue);
        expect(p1.tiles.single.renderStaticTile, isFalse);
        expect(p1.tiles.single.renderCompleteTile, isTrue);
        expect(p1.tiles.single.copySource, SpotTileCopySource.completeTile);

        final p2 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [prop],
          hasStaticShadowCasters: false,
          staticSignature: 0,
        );
        expect(p2.hasRefreshes, isFalse);
        expect(p2.tiles.single.copySource, SpotTileCopySource.completeTile);
      },
    );

    test(
      'transition from static+dynamic to dynamic-only refreshes completeTile to drop removed static shadow',
      () {
        final spot = _spot();
        final prop = _dynamicItem();

        cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 11,
        );
        // Static casters vanish while dynamic prop stays still.
        final p2 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [prop],
          hasStaticShadowCasters: false,
          staticSignature: 0,
        );
        expect(p2.tiles.single.renderStaticTile, isFalse);
        expect(p2.tiles.single.renderCompleteTile, isTrue);
        expect(p2.tiles.single.copySource, SpotTileCopySource.completeTile);
      },
    );

    test(
      'empty caster set emits SpotTileCopySource.none with zero refreshes',
      () {
        final spot = _spot();
        final p = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: const [],
          hasStaticShadowCasters: false,
          staticSignature: 0,
        );
        expect(p.hasRefreshes, isFalse);
        expect(p.tiles.single.copySource, SpotTileCopySource.none);
      },
    );

    test(
      'dynamic caster with MeshDrawSelector refreshes completeTile each frame while keeping staticTile',
      () {
        final spot = _spot();
        final dynamicSelective = _dynamicItem()
          ..drawSource = _StubMeshDrawSource((_) => MeshDrawSelection.all);

        final p1 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [dynamicSelective],
          hasStaticShadowCasters: true,
          staticSignature: 5,
        );
        expect(p1.tiles.single.renderStaticTile, isTrue);
        expect(p1.tiles.single.renderCompleteTile, isTrue);

        final p2 = cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [dynamicSelective],
          hasStaticShadowCasters: true,
          staticSignature: 5,
        );
        expect(p2.tiles.single.renderStaticTile, isFalse);
        expect(p2.tiles.single.renderCompleteTile, isTrue);
      },
    );

    test('matrix float tolerance (1e-6) avoids jitter rebuilds without drift', () {
      final spot = _spot();
      final prop = _dynamicItem();
      final baseFrame = collectSpotShadows([spot])!;
      final baseMatrix = baseFrame.matrices.single.clone();

      cache.plan(
        spotShadows: baseFrame,
        items: [prop],
        hasStaticShadowCasters: true,
        staticSignature: 1,
      );

      // Perturbation within 1e-6 (5e-7) reuses cached tiles and preserves baseMatrix.
      final jittered = baseMatrix.clone();
      jittered.storage[12] += 5e-7;
      final jitterFrame = SpotShadowFrame(
        casters: [spot],
        matrices: [jittered],
        tileResolution: baseFrame.tileResolution,
        depthBias: baseFrame.depthBias,
        normalBias: baseFrame.normalBias,
        softness: baseFrame.softness,
        casterFaces: baseFrame.casterFaces,
        casterChannelMasks: baseFrame.casterChannelMasks,
      );
      final pJitter = cache.plan(
        spotShadows: jitterFrame,
        items: [prop],
        hasStaticShadowCasters: true,
        staticSignature: 1,
      );
      expect(pJitter.hasRefreshes, isFalse);
      expect(pJitter.tiles.single.matrix.storage[12], baseMatrix.storage[12]);

      // Perturbation past 1e-6 (5e-6) invalidates both tiles and updates matrix.
      final moved = baseMatrix.clone();
      moved.storage[12] += 5e-6;
      final movedFrame = SpotShadowFrame(
        casters: [spot],
        matrices: [moved],
        tileResolution: baseFrame.tileResolution,
        depthBias: baseFrame.depthBias,
        normalBias: baseFrame.normalBias,
        softness: baseFrame.softness,
        casterFaces: baseFrame.casterFaces,
        casterChannelMasks: baseFrame.casterChannelMasks,
      );
      final pMoved = cache.plan(
        spotShadows: movedFrame,
        items: [prop],
        hasStaticShadowCasters: true,
        staticSignature: 1,
      );
      expect(pMoved.tiles.single.renderStaticTile, isTrue);
      expect(pMoved.tiles.single.renderCompleteTile, isTrue);
      expect(pMoved.tiles.single.matrix.storage[12], moved.storage[12]);
    });

    test(
      'tileResolution, shadowCasterFaces, and shadowCasterChannelMask changes invalidate cache',
      () {
        final spot = _spot(resolution: 512);
        final prop = _dynamicItem();

        SpotShadowCachePlan runPlan({int? resolutionOverride}) => cache.plan(
          spotShadows: collectSpotShadows([spot])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 9,
          tileResolution: resolutionOverride,
        );

        expect(runPlan().hasRefreshes, isTrue);
        expect(runPlan().hasRefreshes, isFalse);

        // Atlas resolution override (e.g. directional cascade takes over atlas).
        final pRes = runPlan(resolutionOverride: 1024);
        expect(pRes.tileResolution, 1024);
        expect(pRes.tiles.single.renderStaticTile, isTrue);
        expect(pRes.tiles.single.renderCompleteTile, isTrue);
        expect(runPlan(resolutionOverride: 1024).hasRefreshes, isFalse);

        // Caster faces change.
        spot.light.shadowCasterFaces = ShadowCasterFaces.back;
        final pFaces = runPlan(resolutionOverride: 1024);
        expect(pFaces.tiles.single.renderStaticTile, isTrue);
        expect(pFaces.tiles.single.renderCompleteTile, isTrue);
        expect(runPlan(resolutionOverride: 1024).hasRefreshes, isFalse);

        // Caster channel mask change.
        spot.light.shadowCasterChannelMask = 0x0F;
        final pMask = runPlan(resolutionOverride: 1024);
        expect(pMask.tiles.single.renderStaticTile, isTrue);
        expect(pMask.tiles.single.renderCompleteTile, isTrue);
      },
    );

    test(
      'dynamic casters on unrelated light channels do not invalidate spot completeTile',
      () {
        final spotChannel1 = _spot(casterChannelMask: 0x01);
        final propChannel1 = _dynamicItem(
          channelMask: 0x01,
          position: Vector3(0, 1, 0),
        );
        final propChannel2 = _dynamicItem(
          channelMask: 0x02,
          position: Vector3(1, 1, 0),
        );
        final items = [propChannel1, propChannel2];

        cache.plan(
          spotShadows: collectSpotShadows([spotChannel1])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 3,
        );

        // Move propChannel2 (outside spotChannel1's mask): zero refreshes!
        propChannel2.worldTransform.setTranslation(Vector3(5, 2, -3));
        final p = cache.plan(
          spotShadows: collectSpotShadows([spotChannel1])!,
          items: items,
          hasStaticShadowCasters: true,
          staticSignature: 3,
        );
        expect(p.hasRefreshes, isFalse);
      },
    );

    test(
      'component identity keying survives slot reordering, prunes removed spots, and honors cacheStaticShadows=false',
      () {
        final spotA = _spot(position: Vector3(-2, 5, 0));
        final spotB = _spot(position: Vector3(2, 5, 0));
        final prop = _dynamicItem();

        final p1 = cache.plan(
          spotShadows: collectSpotShadows([spotA, spotB])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 1,
        );
        expect(p1.staticRefreshCount, 2);
        expect(p1.completeRefreshCount, 2);
        expect(cache.entryCount, 2);

        // Reorder [spotB, spotA]: both entries hit without thrashing!
        final pSwapped = cache.plan(
          spotShadows: collectSpotShadows([spotB, spotA])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 1,
        );
        expect(pSwapped.hasRefreshes, isFalse);
        expect(pSwapped.tiles[0].component, same(spotB));
        expect(pSwapped.tiles[1].component, same(spotA));

        // Opt spotB out of caching: spotB becomes uncached and its entry is pruned.
        spotB.light.cacheStaticShadows = false;
        final pOptOut = cache.plan(
          spotShadows: collectSpotShadows([spotB, spotA])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 1,
        );
        expect(pOptOut.tiles[0].cached, isFalse);
        expect(pOptOut.tiles[1].cached, isTrue);
        expect(pOptOut.tiles[1].renderStaticTile, isFalse);
        expect(pOptOut.tiles[1].renderCompleteTile, isFalse);
        expect(cache.entryCount, 1);
        expect(cache.entryFor(spotB), isNull);
        expect(cache.entryFor(spotA), isNotNull);

        // Remove spotA from active casters: entry is pruned.
        cache.plan(
          spotShadows: collectSpotShadows([spotB])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 1,
        );
        expect(cache.entryCount, 0);

        // Re-add spotA, then dispose() clears everything.
        cache.plan(
          spotShadows: collectSpotShadows([spotA])!,
          items: [prop],
          hasStaticShadowCasters: true,
          staticSignature: 1,
        );
        expect(cache.entryCount, 1);
        cache.dispose();
        expect(cache.entryCount, 0);
        expect(cache.hasTextures, isFalse);
      },
    );
  });

  group('ShadowPass with SpotShadowCachePlan (GPU-gated)', () {
    TestWidgetsFlutterBinding.ensureInitialized();

    test(
      'executes cached spot static and complete tiles across frames',
      () async {
        if (!gpuAvailable()) return;
        await Scene.initializeStaticResources();

        final scene = Scene()
          ..add(
            Node(
              name: 'floor',
              mesh: Mesh(
                CuboidGeometry(Vector3(4, 0.2, 4)),
                PhysicallyBasedMaterial(),
              ),
            )..shadowStatic = true,
          )
          ..add(
            Node(
              name: 'prop',
              localTransform: Matrix4.translation(Vector3(0, 1, 0)),
              mesh: Mesh(
                CuboidGeometry(Vector3.all(0.5)),
                PhysicallyBasedMaterial(),
              ),
            ),
          );
        final spotNode = Node(
          name: 'spot',
          localTransform: Matrix4.translation(Vector3(0, 5, 0)),
        );
        final spotComp = SpotLightComponent(
          SpotLight(
            direction: Vector3(0, -1, 0),
            range: 15,
            castsShadow: true,
            shadowMapResolution: 128,
          ),
        );
        spotNode.addComponent(spotComp);
        scene.add(spotNode);

        // Sync renderScene items via one small render frame.
        final recorder = ui.PictureRecorder();
        scene.render(
          PerspectiveCamera(position: Vector3(0, 2, 6)),
          ui.Canvas(recorder),
          viewport: const ui.Rect.fromLTWH(0, 0, 16, 16),
        );
        recorder.endRecording().dispose();

        final spotCache = SpotShadowCache();
        final transients = createFrameTransients(rendererSubmissions);
        final pool = TransientTexturePool();
        addTearDown(() {
          spotCache.dispose();
          pool.clear();
        });

        int runShadowPassFrame(int staticSig) {
          pool.beginFrame();
          transients.beginFrame();
          final spotFrame = collectSpotShadows([spotComp])!;
          final plan = spotCache.plan(
            spotShadows: spotFrame,
            items: scene.renderScene.items,
            hasStaticShadowCasters: true,
            staticSignature: staticSig,
          );
          final viewStats = RenderViewStats(
            viewIndex: 0,
            width: 128,
            height: 128,
            offscreen: false,
          );
          final graph = RenderGraph()
            ..addPass(
              ShadowPass(
                renderScene: scene.renderScene,
                tileResolution: spotFrame.tileResolution,
                cameraPosition: Vector3(0, 2, 6),
                spotShadows: spotFrame,
                spotCachePlan: plan,
              ),
            );
          graph.execute(
            texturePool: pool,
            transientsBuffer: transients,
            stats: viewStats,
          );
          return viewStats.passes.single.counters.draws;
        }

        // Frame 1: renders staticTile + completeTile (copy + dynamic draw) + atlas copy.
        final drawsFrame1 = runShadowPassFrame(1);
        expect(spotCache.hasTextures, isTrue);
        expect(drawsFrame1, greaterThanOrEqualTo(3));

        // Frame 2 (stationary): only 1 fullscreen quad copy into the atlas!
        final drawsFrame2 = runShadowPassFrame(1);
        expect(drawsFrame2, 1);

        spotCache.dispose();
        expect(spotCache.hasTextures, isFalse);
        scene.dispose();
      },
    );

    test(
      'Scene end-to-end preserves directional and spot shadow caches across zero-intensity frames',
      () async {
        if (!gpuAvailable()) return;
        await Scene.initializeStaticResources();

        final scene = Scene();
        addTearDown(scene.dispose);

        final sun = DirectionalLight(
          direction: Vector3(0.2, -1.0, -0.3).normalized(),
          intensity: 2.0,
          castsShadow: true,
          cacheStaticShadows: true,
          shadowMapResolution: 128,
        );
        scene.directionalLight = sun;

        final staticFloor = Node(
          mesh: Mesh(
            CuboidGeometry(Vector3(4, 0.2, 4)),
            PhysicallyBasedMaterial(),
          ),
        )..shadowStatic = true;
        scene.add(staticFloor);

        final spotLight = SpotLight(
          direction: Vector3(0, -1, 0),
          intensity: 3.0,
          range: 15,
          castsShadow: true,
          cacheStaticShadows: true,
          shadowMapResolution: 128,
        );
        final spotComp = SpotLightComponent(spotLight);
        final spotNode = Node(
          localTransform: Matrix4.translation(Vector3(0, 4, 0)),
        )..addComponent(spotComp);
        scene.add(spotNode);

        void renderOneFrame() {
          final recorder = ui.PictureRecorder();
          scene.render(
            PerspectiveCamera(position: Vector3(0, 2, 6)),
            ui.Canvas(recorder),
            viewport: const ui.Rect.fromLTWH(0, 0, 32, 32),
          );
          recorder.endRecording().dispose();
        }

        // Frame 1 warms both caches.
        renderOneFrame();
        expect(scene.debugHeldHistoryTargets, contains('shadowCache'));
        expect(scene.debugHeldHistoryTargets, contains('spotShadowCache'));
        expect(scene.isShadowCasterGranted(spotComp), isTrue);

        // Drop both lights to intensity 0: shadow caster slots are skipped,
        // while warm cache textures remain resident for instant wake-up.
        sun.intensity = 0.0;
        spotLight.intensity = 0.0;
        renderOneFrame();
        expect(scene.debugHeldHistoryTargets, contains('shadowCache'));
        expect(scene.debugHeldHistoryTargets, contains('spotShadowCache'));
        expect(scene.isShadowCasterGranted(spotComp), isFalse);

        // Explicitly disabling castsShadow releases the caches.
        sun.castsShadow = false;
        spotLight.castsShadow = false;
        renderOneFrame();
        expect(scene.debugHeldHistoryTargets, isNot(contains('shadowCache')));
        expect(
          scene.debugHeldHistoryTargets,
          isNot(contains('spotShadowCache')),
        );
      },
    );
  });
}
