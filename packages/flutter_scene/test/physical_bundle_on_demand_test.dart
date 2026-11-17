/// The physical material bundle loads on first use instead of gating every
/// scene's first frame. A material that starts needing it asks for it, a
/// failure is reported once and not retried every frame, and an explicit
/// preload retries.
library;

import 'package:flutter/foundation.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
// ignore: implementation_imports
import 'package:flutter_scene/src/material/physical_material.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/material/physical_material_variant.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/render/render_scene.dart';
import 'package:flutter_scene/scene.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

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
  }) {
    throw UnsupportedError('Stub geometry is not renderable');
  }
}

/// The web fetches generated assets over HTTP, bypassing the test asset bundle, and a widget test never completes the request.
const String _webFetch =
    'The web fetches generated assets over HTTP, bypassing the test asset bundle, and a widget test never completes the request.';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late List<FlutterErrorDetails> reported;
  late FlutterExceptionHandler? previousOnError;

  setUp(() {
    PhysicallyBasedMaterial.autoloadPhysicalResources = true;
    resetPhysicalMaterialResourcesForTesting();
    reported = [];
    previousOnError = FlutterError.onError;
    FlutterError.onError = reported.add;
  });

  tearDown(() {
    PhysicallyBasedMaterial.autoloadPhysicalResources = true;
    FlutterError.onError = previousOnError;
    resetPhysicalMaterialResourcesForTesting();
  });

  test('a plain material does not ask for the bundle', () {
    final material = PhysicallyBasedMaterial()
      ..metallicFactor = 0.5
      ..roughnessFactor = 0.3;
    expect(physicalMaterialResourcesPending, isFalse);
    // ignore: invalid_use_of_internal_member
    expect(material.awaitsDeferredResources, isFalse);
  });

  test('an extension asks for the bundle as soon as it is set', () async {
    final material = PhysicallyBasedMaterial()..clearcoat = 0.8;
    expect(physicalMaterialResourcesPending, isTrue);
    // ignore: invalid_use_of_internal_member
    expect(material.awaitsDeferredResources, isTrue);
    expect(physicalMaterialResourcesLoad, isNotNull);

    // The test bundle has no generated assets, so the load fails.
    await physicalMaterialResourcesLoad;
    expect(physicalMaterialResourcesPending, isFalse);
    expect(physicalMaterialResourcesReady, isFalse);
    expect(physicalMaterialResourcesLoad, isNull);
  }, skip: kIsWeb ? _webFetch : null);

  test('a shadow catcher asks for the bundle and draws nothing without '
      'it', () async {
    final catcher = ShadowCatcherMaterial();
    expect(physicalMaterialResourcesPending, isTrue);
    // ignore: invalid_use_of_internal_member
    expect(catcher.drawsNothing, isTrue);
    await physicalMaterialResourcesLoad;
    // ignore: invalid_use_of_internal_member
    expect(catcher.drawsNothing, isTrue);
  }, skip: kIsWeb ? _webFetch : null);

  test('swapping in a shadow catcher hides its item before the next '
      'tick', () {
    final renderScene = RenderScene();
    final root = Node()..debugMountInto(renderScene);
    final geometry = _StubGeometry();
    final node = Node(mesh: Mesh(geometry, PhysicallyBasedMaterial()));
    root.add(node);
    root.scenePrePass(0);
    expect(renderScene.items.single.visible, isTrue);

    // A capture right after the swap must not reach the catcher's unprepared
    // shader.
    node.mesh = Mesh(geometry, ShadowCatcherMaterial());
    expect(renderScene.items.single.visible, isFalse);
  });

  test('a failed request is reported once and not retried', () async {
    requestPhysicalMaterialResources();
    await physicalMaterialResourcesLoad;
    await pumpEventQueue();
    expect(reported, hasLength(1));

    // Materials keep asking every frame; none of that reloads or re-reports.
    requestPhysicalMaterialResources();
    expect(physicalMaterialResourcesPending, isFalse);
    PhysicallyBasedMaterial().clearcoat = 1.0;
    expect(physicalMaterialResourcesPending, isFalse);
    await pumpEventQueue();
    expect(reported, hasLength(1));
  }, skip: kIsWeb ? _webFetch : null);

  test('an explicit load retries after a failure and throws it', () async {
    requestPhysicalMaterialResources();
    await physicalMaterialResourcesLoad;
    await expectLater(initializePhysicalMaterialResources(), throwsA(anything));
    expect(physicalMaterialResourcesPending, isFalse);
  }, skip: kIsWeb ? _webFetch : null);

  test('autoloadPhysicalResources = false suppresses eager bundle loads during '
      'mutation and descriptor import', () async {
    PhysicallyBasedMaterial.autoloadPhysicalResources = false;

    final material = PhysicallyBasedMaterial()
      ..clearcoat = 0.8
      ..sheenColor = Vector4(0.3, 0.3, 0.3, 1.0)
      ..transmission = 0.4;
    expect(physicalMaterialResourcesPending, isFalse);
    expect(physicalMaterialResourcesLoad, isNull);

    // Folding physical extensions back onto standard PBR leaves nothing
    // deferred and never started a bundle download.
    material
      ..clearcoat = 0.0
      ..sheenColor = Vector4.zero()
      ..transmission = 0.0;
    expect(physicalMaterialResourcesPending, isFalse);
    // ignore: invalid_use_of_internal_member
    expect(material.awaitsDeferredResources, isFalse);

    // Descriptor-based import also respects autoloadPhysicalResources = false.
    // ignore: invalid_use_of_internal_member
    final imported = await PhysicallyBasedMaterial.fromDescriptor(
      // ignore: invalid_use_of_internal_member
      PhysicalMaterialDescriptor(clearcoat: 0.9, transmission: 0.2),
    );
    expect(physicalMaterialResourcesPending, isFalse);
    // ignore: invalid_use_of_internal_member
    expect(imported.awaitsDeferredResources, isTrue);
  });

  test('autoloadPhysicalResources = false still requests the bundle if an '
      'unfolded physical material reaches draw preparation', () async {
    PhysicallyBasedMaterial.autoloadPhysicalResources = false;
    final material = PhysicallyBasedMaterial()..clearcoat = 0.7;
    expect(physicalMaterialResourcesPending, isFalse);

    // Accessing fragmentShader runs _ensurePreparedVariant() before resolving
    // the fallback base shader (which throws in unit tests without GPU init).
    expect(() => material.fragmentShader, throwsException);
    expect(physicalMaterialResourcesPending, isTrue);
    await physicalMaterialResourcesLoad;
  }, skip: kIsWeb ? _webFetch : null);
}
