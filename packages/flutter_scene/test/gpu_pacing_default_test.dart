import 'package:flutter/foundation.dart';
import 'package:flutter_scene/src/scene.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  tearDown(() => debugDefaultTargetPlatformOverride = null);

  test('two frames in flight by default on Metal platforms and WebGPU', () {
    for (final platform in TargetPlatform.values) {
      debugDefaultTargetPlatformOverride = platform;
      final metal =
          platform == TargetPlatform.iOS || platform == TargetPlatform.macOS;
      expect(
        defaultMaxGpuFramesInFlight,
        metal && !kIsWeb ? 2 : 1,
        reason: platform.name,
      );
      expect(
        resolveDefaultMaxGpuFramesInFlight(
          isWeb: false,
          immediateGpuExecution: false,
          targetPlatform: platform,
        ),
        metal ? 2 : 1,
        reason: 'native ',
      );
      expect(
        resolveDefaultMaxGpuFramesInFlight(
          isWeb: true,
          immediateGpuExecution: true,
          targetPlatform: platform,
        ),
        1,
        reason: 'WebGL2 ',
      );
      expect(
        resolveDefaultMaxGpuFramesInFlight(
          isWeb: true,
          immediateGpuExecution: false,
          targetPlatform: platform,
        ),
        2,
        reason: 'WebGPU ',
      );
    }
  });
}
