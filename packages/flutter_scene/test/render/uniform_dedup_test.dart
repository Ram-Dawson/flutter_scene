// ignore_for_file: implementation_imports

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_scene/src/fmat/fmat_ast.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/material/material_parameters.dart';
import 'package:flutter_scene/src/material/physically_based_material.dart';
import 'package:flutter_scene/src/material/unlit_material.dart';
import 'package:flutter_scene/src/render/frame_transients.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

bool _canCreateDeviceBuffer() {
  try {
    final buffer = gpu.gpuContext.createDeviceBuffer(
      gpu.StorageMode.hostVisible,
      16,
    );
    gpu.gpuHost.releaseBuffer(buffer);
    return true;
  } catch (_) {
    return false;
  }
}

ByteData _patternedBytes(int length, {int seed = 1}) {
  final list = Uint8List(length);
  var state = seed & 0xFF;
  for (var i = 0; i < length; i++) {
    state = ((state * 73) + i + 19) & 0xFF;
    list[i] = state;
  }
  return ByteData.sublistView(list);
}

class _PassthroughWriter implements TransientWriter {
  int emplaceCalls = 0;
  final TransientArena delegate;

  _PassthroughWriter(this.delegate);

  @override
  gpu.BufferView emplace(ByteData bytes) {
    emplaceCalls++;
    return delegate.emplace(bytes);
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final headless = !_canCreateDeviceBuffer();

  TransientArena createArena(
    GpuSubmissionTracker tracker, {
    int alignment = 256,
    int blockLengthInBytes = TransientArena.kDefaultBlockLengthInBytes,
  }) => TransientArena(
    tracker,
    alignment: alignment,
    blockLengthInBytes: blockLengthInBytes,
    headlessForTesting: headless,
  );

  group('TransientArena word-copy staging', () {
    const sizes = <int>[4, 12, 17, 160, 256, 704, 1028];

    test('stages aligned ByteData accurately across all uniform sizes', () {
      final tracker = GpuSubmissionTracker();
      final arena = createArena(tracker, alignment: 256);

      // Pre-fill the open block with a canary pattern after opening it so we
      // can verify staging touches neither preceding nor trailing bytes.
      final warmup = arena.emplace(ByteData(4));
      expect(warmup.offsetInBytes, 0);
      final rawStaging = arena.debugOpenBlockStaging(0);
      for (var i = 4; i < 8192; i++) {
        rawStaging.setUint8(i, 0xCC);
      }

      var previousEnd = 4;
      final priorViews = <(gpu.BufferView, Uint8List)>[];
      for (var idx = 0; idx < sizes.length; idx++) {
        final size = sizes[idx];
        final src = _patternedBytes(size, seed: idx + 11);
        final view = arena.emplace(src);
        expect(view.lengthInBytes, size);
        expect(view.offsetInBytes % 256, 0);

        // Every alignment-padding byte between the previous payload and this
        // aligned offset must remain untouched.
        for (var pad = previousEnd; pad < view.offsetInBytes; pad++) {
          expect(rawStaging.getUint8(pad), 0xCC);
        }
        // Trailing canary byte immediately past [offset + size] is untouched.
        expect(rawStaging.getUint8(view.offsetInBytes + size), 0xCC);

        final staged = arena.debugReadStagedBytes(view);
        final expected = Uint8List.fromList(
          src.buffer.asUint8List(src.offsetInBytes, size),
        );
        expect(staged, expected, reason: 'byte mismatch at size ${size}B');
        priorViews.add((view, expected));
        previousEnd = view.offsetInBytes + size;
      }

      // All earlier slices (including 256B immediately abutting 704B) remain
      // intact after subsequent emplacements.
      for (final (view, expected) in priorViews) {
        expect(arena.debugReadStagedBytes(view), expected);
      }
    });

    test(
      'stages unaligned ByteData subviews (offsetInBytes: 2) accurately',
      () {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 16);

        for (final size in sizes) {
          final backing = Uint8List(size + 8);
          for (var i = 0; i < backing.length; i++) {
            backing[i] = 0xEE;
          }
          final expected = Uint8List(size);
          for (var i = 0; i < size; i++) {
            final value = ((i * 37) + size + 5) & 0xFF;
            backing[2 + i] = value;
            expected[i] = value;
          }

          final unalignedView = ByteData.sublistView(backing, 2, 2 + size);
          expect(unalignedView.offsetInBytes & 3, 2);
          expect(unalignedView.lengthInBytes, size);

          final view = arena.emplace(unalignedView);
          expect(
            arena.debugReadStagedBytes(view),
            expected,
            reason: 'unaligned subview mismatch at size ${size}B',
          );
        }
      },
    );
  });

  group('TransientArena.emplaceDeduplicated', () {
    test('deduplicates 200 identical 160B TextureTransforms and 704B FragInfo '
        'blocks with >99% byte reduction', () {
      for (final blockSize in const <int>[160, 704]) {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 256);
        final TransientWriter writer = arena;
        final payload = _patternedBytes(blockSize, seed: blockSize);

        final first = writer.emplaceDeduplicated(payload);
        expect(first.offsetInBytes, 0);
        expect(first.lengthInBytes, blockSize);
        expect(arena.stagedBytes, blockSize);
        expect(arena.deduplicatedEmplaces, 0);

        for (var draw = 1; draw < 200; draw++) {
          // Also pass a freshly allocated copy with identical bytes to prove
          // deduplication is content-addressed rather than identity-based.
          final copy = ByteData(blockSize);
          for (var b = 0; b < blockSize; b++) {
            copy.setUint8(b, payload.getUint8(b));
          }
          final next = writer.emplaceDeduplicated(copy);
          expect(identical(next, first), isTrue);
          expect(next.offsetInBytes, first.offsetInBytes);
          expect(next.lengthInBytes, blockSize);
        }

        final unoptimizedBytes = 200 * blockSize;
        final reduction = 1.0 - (arena.stagedBytes / unoptimizedBytes);
        expect(arena.stagedBytes, blockSize);
        expect(arena.deduplicatedEmplaces, 199);
        expect(arena.deduplicatedBytes, 199 * blockSize);
        expect(reduction, greaterThan(0.99));
      }
    });

    test(
      'deduplicates unaligned ByteData subviews against aligned ByteData',
      () {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 256);

        final aligned = _patternedBytes(160, seed: 42);
        final padded = Uint8List(166);
        for (var i = 0; i < 160; i++) {
          padded[2 + i] = aligned.getUint8(i);
        }
        final unaligned = ByteData.sublistView(padded, 2, 162);
        expect(unaligned.offsetInBytes, 2);

        final first = arena.emplaceDeduplicated(aligned);
        final second = arena.emplaceDeduplicated(unaligned);
        expect(identical(second, first), isTrue);
        expect(arena.stagedBytes, 160);
        expect(arena.deduplicatedEmplaces, 1);
      },
    );

    test('allocates a new BufferView when any single byte changes', () {
      final tracker = GpuSubmissionTracker();
      final arena = createArena(tracker, alignment: 256);

      // Include a non-multiple-of-4 size (705B) as well as 704B so both 32-bit
      // words and trailing bytes are verified.
      for (final size in const <int>[704, 705]) {
        arena.reset();
        final base = _patternedBytes(size, seed: 7);
        final baseView = arena.emplaceDeduplicated(base);
        expect(baseView.offsetInBytes, 0);

        // Mutate first byte, a middle word byte, and the very last byte.
        final positions = <int>[0, size >> 1, size - 1];
        var expectedStaged = size;
        for (final pos in positions) {
          final mutated = ByteData(size);
          for (var i = 0; i < size; i++) {
            mutated.setUint8(i, base.getUint8(i));
          }
          mutated.setUint8(pos, base.getUint8(pos) ^ 0x01);

          final nextView = arena.emplaceDeduplicated(mutated);
          expect(identical(nextView, baseView), isFalse);
          expect(nextView.offsetInBytes, greaterThan(baseView.offsetInBytes));
          expectedStaged += size;
          expect(arena.stagedBytes, expectedStaged);

          // Re-emplacing the original bytes still finds the original view!
          final hitAgain = arena.emplaceDeduplicated(base);
          expect(identical(hitAgain, baseView), isTrue);
        }
      }
    });

    test(
      'resets cleanly across arena.reset(), beginFrame(), and submissions',
      () {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 256);
        final data = _patternedBytes(704, seed: 99);

        final v1 = arena.emplaceDeduplicated(data);
        final v1Hit = arena.emplaceDeduplicated(data);
        expect(identical(v1Hit, v1), isTrue);
        expect(arena.stagedBytes, 704);
        expect(arena.deduplicatedEmplaces, 1);

        // Explicit reset() clears stats and invalidates the epoch in O(1).
        arena.reset();
        expect(arena.stagedBytes, 0);
        expect(arena.deduplicatedBytes, 0);
        expect(arena.deduplicatedEmplaces, 0);

        // Mutating the source buffer before emplacing in the new epoch stages
        // the new contents rather than returning a stale view.
        final v2 = arena.emplaceDeduplicated(data);
        expect(identical(v2, v1), isFalse);
        expect(arena.stagedBytes, 704);
        expect(arena.deduplicatedEmplaces, 0);

        // Recording a GPU submission seals open blocks and starts a fresh dedup
        // epoch so sealed blocks are never referenced by open-block indices.
        final subId = tracker.record();
        final v3 = arena.emplaceDeduplicated(data);
        expect(identical(v3, v2), isFalse);
        expect(arena.stagedBytes, 1408);

        tracker.complete(subId);
        arena.beginFrame();
        expect(arena.stagedBytes, 0);
        expect(arena.deduplicatedEmplaces, 0);
      },
    );

    test(
      'non-arena TransientWriter implementations fall back to emplace()',
      () {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 256);
        final TransientWriter wrapper = _PassthroughWriter(arena);
        final bytes = _patternedBytes(64, seed: 3);

        final a = wrapper.emplaceDeduplicated(bytes);
        final b = wrapper.emplaceDeduplicated(bytes);
        expect((wrapper as _PassthroughWriter).emplaceCalls, 2);
        expect(a.offsetInBytes, 0);
        expect(b.offsetInBytes, 256);
      },
    );

    test('deduplicates MaterialParameters rawBlock across shared draws', () {
      final tracker = GpuSubmissionTracker();
      final arena = createArena(tracker, alignment: 256);
      final params =
          MaterialParameters.withLayout(
              blockName: 'MaterialParams',
              blockSizeBytes: 64,
              parameters: {
                'tint': (type: FmatType.vec4, offset: 0, sourceColor: false),
                'roughness': (
                  type: FmatType.float_,
                  offset: 16,
                  sourceColor: false,
                ),
              },
            )
            ..setVec4('tint', Vector4(0.3, 0.5, 0.7, 1.0))
            ..setFloat('roughness', 0.4);

      final first = arena.emplaceDeduplicated(params.rawBlock);
      for (var i = 0; i < 50; i++) {
        expect(
          identical(arena.emplaceDeduplicated(params.rawBlock), first),
          isTrue,
        );
      }
      expect(arena.stagedBytes, 64);

      params.setFloat('roughness', 0.8);
      final updated = arena.emplaceDeduplicated(params.rawBlock);
      expect(identical(updated, first), isFalse);
      expect(arena.stagedBytes, 128);
    });
  });

  group('PhysicallyBasedMaterial TextureTransforms caching & invalidation', () {
    test('packs once when clean and shares one BufferView across materials', () {
      final tracker = GpuSubmissionTracker();
      final arena = createArena(tracker, alignment: 256);
      final materials = List<PhysicallyBasedMaterial>.generate(
        30,
        (i) => PhysicallyBasedMaterial()
          ..metallicFactor = (i % 5) * 0.25
          ..roughnessFactor = 0.2 + (i % 4) * 0.2,
      );

      // Simulate 206 draws across 30 materials.
      gpu.BufferView? sharedView;
      for (var draw = 0; draw < 206; draw++) {
        final material = materials[draw % materials.length];
        final bytes = material.packTextureTransformsForTesting();
        expect(bytes.lengthInBytes, 160);
        final view = arena.emplaceDeduplicated(bytes);
        sharedView ??= view;
        expect(identical(view, sharedView), isTrue);
      }

      // Each of the 30 materials packed its 40 floats exactly once (not 206
      // times), and TransientArena staged 160 bytes once across all 206 draws!
      for (final material in materials) {
        expect(material.debugTextureTransformsDirty, isFalse);
        expect(material.debugTextureTransformsPackCount, 1);
      }
      expect(arena.stagedBytes, 160);
      expect(arena.deduplicatedEmplaces, 205);

      // Non-transform scalar/factor mutations do NOT dirty TextureTransforms.
      materials.first
        ..baseColorFactor = Vector4(0.8, 0.2, 0.1, 1.0)
        ..metallicFactor = 0.9
        ..roughnessFactor = 0.1
        ..normalScale = 0.5
        ..occlusionStrength = 0.7;
      expect(materials.first.debugTextureTransformsDirty, isFalse);
      materials.first.packTextureTransformsForTesting();
      expect(materials.first.debugTextureTransformsPackCount, 1);

      // UnlitMaterial transform getter/setter also round-trips cleanly.
      final unlit = UnlitMaterial()
        ..baseColorTextureTransform = TextureTransform(rotation: 0.25)
        ..baseColorTextureTexCoord = 1;
      expect(unlit.baseColorTextureTransform.rotation, 0.25);
      expect(unlit.baseColorTextureTexCoord, 1);
    });

    test(
      'invalidates cached TextureTransforms on every transform and texCoord setter',
      () {
        final material = PhysicallyBasedMaterial();

        // Initial pack: identity transforms, UV set 0, gate flag [7] == 0.
        final initial = material.packTextureTransformsForTesting();
        expect(material.debugTextureTransformsDirty, isFalse);
        expect(initial.getFloat32(7 * 4, Endian.host), 0.0);

        final customTransform = TextureTransform(
          offset: Vector2(0.25, -0.5),
          scale: Vector2(2.0, 3.0),
          rotation: math.pi / 3,
        );
        material.baseColorTextureTransform = customTransform;
        expect(material.debugTextureTransformsDirty, isTrue);

        final repacked = material.packTextureTransformsForTesting();
        expect(material.debugTextureTransformsDirty, isFalse);
        expect(material.debugTextureTransformsPackCount, 2);
        expect(repacked.getFloat32(0 * 4, Endian.host), closeTo(0.25, 1e-6));
        expect(repacked.getFloat32(1 * 4, Endian.host), closeTo(-0.5, 1e-6));
        expect(repacked.getFloat32(2 * 4, Endian.host), closeTo(2.0, 1e-6));
        expect(repacked.getFloat32(3 * 4, Endian.host), closeTo(3.0, 1e-6));
        expect(
          repacked.getFloat32(4 * 4, Endian.host),
          closeTo(math.cos(math.pi / 3), 1e-6),
        );
        expect(
          repacked.getFloat32(5 * 4, Endian.host),
          closeTo(math.sin(math.pi / 3), 1e-6),
        );
        // transformedUvs gate flag at float [7] is now 1.0.
        expect(repacked.getFloat32(7 * 4, Endian.host), 1.0);

        // Verify every transform and texCoord setter marks the cache dirty.
        final mutators = <String, void Function(PhysicallyBasedMaterial)>{
          'baseColorTextureTransform': (m) =>
              m.baseColorTextureTransform = TextureTransform(),
          'baseColorTextureTexCoord': (m) => m.baseColorTextureTexCoord = 1,
          'metallicRoughnessTextureTransform': (m) =>
              m.metallicRoughnessTextureTransform = TextureTransform(
                scale: Vector2.all(2),
              ),
          'metallicRoughnessTextureTexCoord': (m) =>
              m.metallicRoughnessTextureTexCoord = 1,
          'normalTextureTransform': (m) =>
              m.normalTextureTransform = TextureTransform(rotation: 0.4),
          'normalTextureTexCoord': (m) => m.normalTextureTexCoord = 1,
          'emissiveTextureTransform': (m) => m.emissiveTextureTransform =
              TextureTransform(offset: Vector2(0.1, 0.2)),
          'emissiveTextureTexCoord': (m) => m.emissiveTextureTexCoord = 1,
          'occlusionTextureTransform': (m) => m.occlusionTextureTransform =
              TextureTransform(scale: Vector2(1, 2)),
          'occlusionTextureTexCoord': (m) => m.occlusionTextureTexCoord = 1,
          'lightmapTextureTransform': (m) =>
              m.lightmapTextureTransform = TextureTransform(),
          'lightmapTextureTexCoord': (m) => m.lightmapTextureTexCoord = 0,
          'specularTextureTransform': (m) =>
              m.specularTextureTransform = TextureTransform(),
          'specularTextureTexCoord': (m) => m.specularTextureTexCoord = 1,
          'specularColorTextureTransform': (m) =>
              m.specularColorTextureTransform = TextureTransform(),
          'specularColorTextureTexCoord': (m) =>
              m.specularColorTextureTexCoord = 1,
          'clearcoatTextureTransform': (m) =>
              m.clearcoatTextureTransform = TextureTransform(),
          'clearcoatTextureTexCoord': (m) => m.clearcoatTextureTexCoord = 1,
          'clearcoatRoughnessTextureTransform': (m) =>
              m.clearcoatRoughnessTextureTransform = TextureTransform(),
          'clearcoatRoughnessTextureTexCoord': (m) =>
              m.clearcoatRoughnessTextureTexCoord = 1,
          'clearcoatNormalTextureTransform': (m) =>
              m.clearcoatNormalTextureTransform = TextureTransform(),
          'clearcoatNormalTextureTexCoord': (m) =>
              m.clearcoatNormalTextureTexCoord = 1,
          'sheenColorTextureTransform': (m) =>
              m.sheenColorTextureTransform = TextureTransform(),
          'sheenColorTextureTexCoord': (m) => m.sheenColorTextureTexCoord = 1,
          'sheenRoughnessTextureTransform': (m) =>
              m.sheenRoughnessTextureTransform = TextureTransform(),
          'sheenRoughnessTextureTexCoord': (m) =>
              m.sheenRoughnessTextureTexCoord = 1,
          'transmissionTextureTransform': (m) =>
              m.transmissionTextureTransform = TextureTransform(),
          'transmissionTextureTexCoord': (m) =>
              m.transmissionTextureTexCoord = 1,
          'diffuseTransmissionTextureTransform': (m) =>
              m.diffuseTransmissionTextureTransform = TextureTransform(),
          'diffuseTransmissionTextureTexCoord': (m) =>
              m.diffuseTransmissionTextureTexCoord = 1,
          'diffuseTransmissionColorTextureTransform': (m) =>
              m.diffuseTransmissionColorTextureTransform = TextureTransform(),
          'diffuseTransmissionColorTextureTexCoord': (m) =>
              m.diffuseTransmissionColorTextureTexCoord = 1,
          'thicknessTextureTransform': (m) =>
              m.thicknessTextureTransform = TextureTransform(),
          'thicknessTextureTexCoord': (m) => m.thicknessTextureTexCoord = 1,
          'iridescenceTextureTransform': (m) =>
              m.iridescenceTextureTransform = TextureTransform(),
          'iridescenceTextureTexCoord': (m) => m.iridescenceTextureTexCoord = 1,
          'iridescenceThicknessTextureTransform': (m) =>
              m.iridescenceThicknessTextureTransform = TextureTransform(),
          'iridescenceThicknessTextureTexCoord': (m) =>
              m.iridescenceThicknessTextureTexCoord = 1,
          'anisotropyTextureTransform': (m) =>
              m.anisotropyTextureTransform = TextureTransform(),
          'anisotropyTextureTexCoord': (m) => m.anisotropyTextureTexCoord = 1,
        };

        for (final entry in mutators.entries) {
          material.packTextureTransformsForTesting();
          expect(material.debugTextureTransformsDirty, isFalse);
          entry.value(material);
          expect(
            material.debugTextureTransformsDirty,
            isTrue,
            reason: '${entry.key} did not mark TextureTransforms dirty',
          );
        }
      },
    );

    test(
      'repacks UBO on in-place TextureTransform field mutations without setter calls (UnlitMaterial & PhysicallyBasedMaterial)',
      () {
        final tracker = GpuSubmissionTracker();
        final arena = createArena(tracker, alignment: 256);

        // 1) UnlitMaterial in-place mutation (e.g. cloud UV scrolling via t.offset.x = ...)
        final unlit = UnlitMaterial();
        final unlitInitialBytes = unlit.packTextureTransformForTesting();
        final unlitView1 = arena.emplaceDeduplicated(unlitInitialBytes);
        final unlitStaged1 = Uint8List.fromList(
          arena.debugReadStagedBytes(unlitView1),
        );
        expect(unlit.debugTextureTransformPackCount, 1);
        expect(unlit.debugTextureTransformDirty, isFalse);
        expect(arena.stagedBytes, 32);

        // Steady bind without changes does not repack or restage.
        final unlitView1Repeat = arena.emplaceDeduplicated(
          unlit.packTextureTransformForTesting(),
        );
        expect(unlit.debugTextureTransformPackCount, 1);
        expect(identical(unlitView1Repeat, unlitView1), isTrue);
        expect(arena.stagedBytes, 32);

        // Mutate offset.x in place without invoking baseColorTextureTransform setter.
        unlit.baseColorTextureTransform.offset.x += 0.25;
        expect(unlit.debugTextureTransformDirty, isTrue);

        final unlitRepacked = unlit.packTextureTransformForTesting();
        final unlitView2 = arena.emplaceDeduplicated(unlitRepacked);
        final unlitStaged2 = Uint8List.fromList(
          arena.debugReadStagedBytes(unlitView2),
        );
        expect(unlit.debugTextureTransformPackCount, 2);
        expect(unlit.debugTextureTransformDirty, isFalse);
        expect(identical(unlitView2, unlitView1), isFalse);
        expect(unlitStaged2, isNot(equals(unlitStaged1)));
        expect(unlitRepacked.getFloat32(0, Endian.host), closeTo(0.25, 1e-6));
        expect(arena.stagedBytes, 64);

        // Subsequent steady binds with irrational double values do not re-pack.
        unlit.baseColorTextureTransform
          ..scale.y = -1.0
          ..rotation = math.pi / 7;
        expect(unlit.debugTextureTransformDirty, isTrue);
        final unlitView3 = arena.emplaceDeduplicated(
          unlit.packTextureTransformForTesting(),
        );
        expect(unlit.debugTextureTransformPackCount, 3);
        expect(arena.stagedBytes, 96);
        for (var i = 0; i < 10; i++) {
          final steady = arena.emplaceDeduplicated(
            unlit.packTextureTransformForTesting(),
          );
          expect(identical(steady, unlitView3), isTrue);
        }
        expect(unlit.debugTextureTransformPackCount, 3);

        // 2) PhysicallyBasedMaterial in-place mutations across texture slots
        arena.reset();
        final pbr = PhysicallyBasedMaterial();
        final pbrInitial = pbr.packTextureTransformsForTesting();
        final pbrView1 = arena.emplaceDeduplicated(pbrInitial);
        final pbrStaged1 = Uint8List.fromList(
          arena.debugReadStagedBytes(pbrView1),
        );
        expect(pbr.debugTextureTransformsPackCount, 1);
        expect(pbr.debugTextureTransformsDirty, isFalse);
        expect(pbrInitial.getFloat32(7 * 4, Endian.host), 0.0);
        expect(arena.stagedBytes, 160);

        // Steady bind before mutation does not repack.
        expect(
          identical(
            arena.emplaceDeduplicated(pbr.packTextureTransformsForTesting()),
            pbrView1,
          ),
          isTrue,
        );
        expect(pbr.debugTextureTransformsPackCount, 1);

        // In-place baseColorTextureTransform.offset.x += 0.25 (no setter)
        pbr.baseColorTextureTransform.offset.x += 0.25;
        expect(pbr.debugTextureTransformsDirty, isTrue);
        final pbrAfterBase = pbr.packTextureTransformsForTesting();
        final pbrView2 = arena.emplaceDeduplicated(pbrAfterBase);
        final pbrStaged2 = Uint8List.fromList(
          arena.debugReadStagedBytes(pbrView2),
        );
        expect(pbr.debugTextureTransformsPackCount, 2);
        expect(pbr.debugTextureTransformsDirty, isFalse);
        expect(identical(pbrView2, pbrView1), isFalse);
        expect(pbrStaged2, isNot(equals(pbrStaged1)));
        expect(
          pbrAfterBase.getFloat32(0 * 4, Endian.host),
          closeTo(0.25, 1e-6),
        );
        // transformedUvs gate flag at float [7] flips to 1.0!
        expect(pbrAfterBase.getFloat32(7 * 4, Endian.host), 1.0);
        expect(arena.stagedBytes, 320);

        // Steady bind after baseColor mutation does not repack.
        expect(
          identical(
            arena.emplaceDeduplicated(pbr.packTextureTransformsForTesting()),
            pbrView2,
          ),
          isTrue,
        );
        expect(pbr.debugTextureTransformsPackCount, 2);

        // In-place emissiveTextureTransform.rotation (irrational double)
        pbr.emissiveTextureTransform.rotation = math.pi / 4;
        expect(pbr.debugTextureTransformsDirty, isTrue);
        final pbrAfterEmissive = pbr.packTextureTransformsForTesting();
        final pbrView3 = arena.emplaceDeduplicated(pbrAfterEmissive);
        final pbrStaged3 = Uint8List.fromList(
          arena.debugReadStagedBytes(pbrView3),
        );
        expect(pbr.debugTextureTransformsPackCount, 3);
        expect(identical(pbrView3, pbrView2), isFalse);
        expect(pbrStaged3, isNot(equals(pbrStaged2)));
        expect(
          pbrAfterEmissive.getFloat32((24 + 4) * 4, Endian.host),
          closeTo(math.cos(math.pi / 4), 1e-6),
        );
        expect(
          pbrAfterEmissive.getFloat32((24 + 5) * 4, Endian.host),
          closeTo(math.sin(math.pi / 4), 1e-6),
        );
        expect(arena.stagedBytes, 480);

        // In-place normalTextureTransform.scale.y
        pbr.normalTextureTransform.scale.y = 2.5;
        expect(pbr.debugTextureTransformsDirty, isTrue);
        final pbrAfterNormal = pbr.packTextureTransformsForTesting();
        final pbrView4 = arena.emplaceDeduplicated(pbrAfterNormal);
        expect(pbr.debugTextureTransformsPackCount, 4);
        expect(identical(pbrView4, pbrView3), isFalse);
        expect(
          pbrAfterNormal.getFloat32((16 + 3) * 4, Endian.host),
          closeTo(2.5, 1e-6),
        );
        expect(arena.stagedBytes, 640);

        // 10 consecutive steady binds with non-float32-exact doubles must NOT repack.
        for (var i = 0; i < 10; i++) {
          final steady = arena.emplaceDeduplicated(
            pbr.packTextureTransformsForTesting(),
          );
          expect(identical(steady, pbrView4), isTrue);
        }
        expect(pbr.debugTextureTransformsPackCount, 4);
        expect(arena.stagedBytes, 640);
      },
    );
  });
}
