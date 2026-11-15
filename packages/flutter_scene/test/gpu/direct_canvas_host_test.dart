// ignore_for_file: implementation_imports
import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/gpu/stub/_gpu.dart' as stub;
import 'package:flutter_test/flutter_test.dart';

import '../support/gpu_available.dart';

void main() {
  group('GpuHost direct DOM canvas hooks (active backend)', () {
    test('exposes the shared direct-present canvas element id', () {
      expect(gpu.GpuHost.kDirectPresentCanvasId, 'flutter-scene-webgpu-canvas');
    });

    test(
      'canPresentToDomCanvas is false without initializing a GPU context',
      () {
        // Querying direct-present readiness before initializeGpuBackend() (or
        // where Impeller is disabled under headless flutter_test) must never
        // throw a StateError or construct a GPU context.
        expect(gpu.gpuHost.canPresentToDomCanvas, isFalse);
      },
    );

    test('exposes presentToDomCanvas with the expected signature', () {
      final bool Function(gpu.Texture, {required ui.Rect logicalArea}) fn =
          gpu.gpuHost.presentToDomCanvas;
      expect(fn, isNotNull);
    });

    test(
      'presentToDomCanvas returns false cleanly when no DOM canvas is active',
      () async {
        gpu.Texture? texture;
        try {
          await gpu.initializeGpuBackend();
          texture = gpu.gpuContext.createTexture(
            gpu.StorageMode.hostVisible,
            4,
            4,
          );
        } catch (_) {
          // Headless VM test runs without Impeller cannot allocate GPU
          // textures; browser runs and Impeller-enabled runs exercise the call.
          texture = null;
        }
        if (texture != null) {
          addTearDown(() => gpu.gpuHost.releaseTexture(texture!));
          expect(
            gpu.gpuHost.presentToDomCanvas(
              texture,
              logicalArea: const ui.Rect.fromLTWH(0, 0, 100, 80),
            ),
            isFalse,
          );
        }
      },
    );
  });

  group('GpuHost direct DOM canvas hooks (stub backend)', () {
    test('exposes matching canvas id and safe false defaults', () {
      expect(
        stub.GpuHost.kDirectPresentCanvasId,
        'flutter-scene-webgpu-canvas',
      );
      expect(stub.gpuHost.canPresentToDomCanvas, isFalse);
      final bool Function(stub.Texture, {required ui.Rect logicalArea}) fn =
          stub.gpuHost.presentToDomCanvas;
      expect(fn, isNotNull);
    });

    test(
      'Scene.canRenderDirectToDomCanvas is false when no DOM canvas is active',
      () {
        if (!gpuAvailable()) return;
        final scene = Scene();
        addTearDown(scene.dispose);
        expect(scene.canRenderDirectToDomCanvas, isFalse);
      },
    );
  });
}
