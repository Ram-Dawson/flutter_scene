// Guards the dead-code elimination guarantee documented for <noise.glsl>:
// NoiseValue2/NoiseValue3 and NoiseHash2/NoiseHash3 use integer hashing only,
// so compiling a material that includes <noise.glsl> and calls only those
// helpers must strip the top-level 2,048 floats of lookup tables from the
// emitted OpenGL ES shader instead of passing them into ANGLE's D3D compiler.

@TestOn('vm')
library;

import 'dart:io';

import 'package:flutter_gpu_shaders/environment.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/fmat/fmat.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/generated_assets/build_engine_assets.dart'
    show engineGlesLanguageVersion;
import 'package:flutter_test/flutter_test.dart';

const _tableSymbols = [
  'noise_gradients2D',
  'noise_gradients3D',
  'noise_randVecs2D',
  'noise_randVecs3D',
];

final _lookupArrayPattern = RegExp(r'\[(?:256|512|1024)\]');

Future<String> _compileUnlitNoiseGles({
  required String name,
  required String surfaceBody,
}) async {
  final fmatSource =
      '''
material {
  name: "$name",
  shading_model: unlit,
}

fragment {
  #include <noise.glsl>

  void Surface(inout MaterialInputs material) {
$surfaceBody
  }
}
''';
  final compiled = compileFmat(fmatSource, fileName: '$name.fmat');
  final glslSource = emitFragmentGlsl(compiled.material);
  final impellerc = await findImpellerC();
  final temp = Directory.systemTemp.createTempSync('noise_shader_dce');
  try {
    final input = File.fromUri(temp.uri.resolve('$name.frag'))
      ..writeAsStringSync(glslSource);
    final output = File.fromUri(temp.uri.resolve('$name.gles'));
    final spirv = temp.uri.resolve('$name.spirv').toFilePath();
    final result = await Process.run(impellerc.toFilePath(), [
      '--opengl-es',
      '--input-type=frag',
      '--input=${input.path}',
      '--sl=${output.path}',
      '--spirv=$spirv',
      '--define=IMPELLER_TARGET_OPENGLES',
      '--include=${Directory.current.uri.resolve('shaders/').toFilePath()}',
      '--include=${impellerc.resolve('./shader_lib').toFilePath()}',
      '--gles-language-version=$engineGlesLanguageVersion',
    ]);
    expect(
      result.exitCode,
      0,
      reason: '$name\n${result.stdout}\n${result.stderr}',
    );
    return output.readAsStringSync();
  } finally {
    temp.deleteSync(recursive: true);
  }
}

void _expectNoLookupTables(String glesSource, {required String reason}) {
  for (final symbol in _tableSymbols) {
    expect(glesSource, isNot(contains(symbol)), reason: reason);
  }
  expect(_lookupArrayPattern.hasMatch(glesSource), isFalse, reason: reason);
}

void main() {
  test(
    'NoiseValue2 and NoiseValue3 strip unused <noise.glsl> lookup tables',
    () async {
      final gles = await _compileUnlitNoiseGles(
        name: 'UnlitValueNoise',
        surfaceBody: '''
    vec3 p = GetWorldPosition();
    float v2 = NoiseValue2(p.xy, 1337);
    float v3 = NoiseValue3(p, 1337);
    material.base_color = vec4(vec3(0.5 + 0.25 * (v2 + v3)), 1.0);''',
      );
      _expectNoLookupTables(
        gles,
        reason: 'NoiseValue2/NoiseValue3 must not retain <noise.glsl> tables.',
      );
    },
  );

  test(
    'NoiseHash2 and NoiseHash3 strip unused <noise.glsl> lookup tables',
    () async {
      final gles = await _compileUnlitNoiseGles(
        name: 'UnlitHashNoise',
        surfaceBody: '''
    vec3 p = GetWorldPosition();
    float h2 = NoiseHash2(ivec2(floor(p.xy)), 1337);
    float h3 = NoiseHash3(ivec3(floor(p)), 1337);
    material.base_color = vec4(vec3(h2 * 0.5 + h3 * 0.5), 1.0);''',
      );
      _expectNoLookupTables(
        gles,
        reason: 'NoiseHash2/NoiseHash3 must not retain <noise.glsl> tables.',
      );
    },
  );

  test('NoiseSimplex3 retains a 256-float lookup table in OpenGL ES', () async {
    final gles = await _compileUnlitNoiseGles(
      name: 'UnlitSimplexNoise',
      surfaceBody: '''
    float s = NoiseSimplex3(GetWorldPosition(), 1337);
    material.base_color = vec4(vec3(0.5 + 0.5 * s), 1.0);''',
    );
    expect(
      gles,
      contains('[256]'),
      reason:
          'NoiseSimplex3 should emit its 256-float gradient table so the DCE '
          'array matcher stays calibrated.',
    );
  });
}
