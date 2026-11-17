// The off-thread mip chain build. Split from mipmap.dart because that file is
// re-exported by `build_hooks.dart`, and a build hook runs on the plain Dart
// VM where `package:flutter` (and so `dart:ui`) cannot be resolved.

import 'dart:typed_data';

import 'package:flutter/foundation.dart' show compute, kIsWeb;

import 'mipmap.dart';
import 'web_transcode_time_slicer.dart';

/// Builds the mip chain for [pixels] on a background isolate (or time-sliced on
/// the main thread on Web), so large textures neither block callers nor starve
/// browser event loop turns during bulk texture loads.
///
/// The synchronous [generateMipChain] stays for the sync realize path, which
/// cannot await.
Future<List<MipLevel>> generateMipChainAsync(
  Uint8List pixels,
  int width,
  int height,
  TextureContent content,
) {
  final input = (
    pixels: pixels,
    width: width,
    height: height,
    content: content,
  );
  if (kIsWeb) {
    return webTranscodeTimeSlicer.run(() => _generateMipChain(input));
  }
  return compute(_generateMipChain, input);
}

/// Isolate entry point. Pure Dart, no GPU.
List<MipLevel> _generateMipChain(
  ({Uint8List pixels, int width, int height, TextureContent content}) input,
) => generateMipChain(input.pixels, input.width, input.height, input.content);
