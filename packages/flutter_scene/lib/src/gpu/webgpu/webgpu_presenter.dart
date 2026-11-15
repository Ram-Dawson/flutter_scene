/// Getting a WebGPU frame into Flutter as a `ui.Image`, synchronously, or
/// presenting it directly into an opted-in on-screen DOM `<canvas>`.
///
/// The `ui.Image` path mirrors the WebGL2 backend (`snapshotTextureSync` in
/// `web/gpu_context.dart`), because `Scene.render` draws the frame inside a
/// synchronous paint callback. The canvas's current texture is rendered,
/// submitted, and transferred out with `transferToImageBitmap`, which takes the
/// submitted contents and gives the canvas a fresh texture for the next frame.
/// When an opted-in DOM `<canvas>` is mounted, [blitToDomCanvasSync] renders
/// the same fullscreen blit directly into that swapchain texture instead.
library;

import 'dart:js_interop';
import 'dart:ui' as ui;
import 'dart:ui_web' as ui_web;

import 'package:web/web.dart' as web;

import 'webgpu_device.dart';
import 'webgpu_interop.dart';

/// Owns the `OffscreenCanvas` frames are handed to Flutter through, and an
/// optional direct presentation path targeting an opted-in on-screen DOM
/// `<canvas>`.
final class WebGpuPresenter {
  WebGpuPresenter(this.device) : _canvas = web.OffscreenCanvas(1, 1) {
    final context = _canvas.getContext('webgpu');
    if (context == null) {
      throw StateError('OffscreenCanvas has no webgpu context.');
    }
    _context = context as GPUCanvasContext;
  }

  /// DOM element id for the opt-in on-screen `<canvas>` used by direct WebGPU
  /// presentation.
  static const String kDirectPresentCanvasId = 'flutter-scene-webgpu-canvas';

  final WebGpuDevice device;
  final web.OffscreenCanvas _canvas;
  late final GPUCanvasContext _context;
  bool _configured = false;

  web.HTMLCanvasElement? _domCanvas;
  GPUCanvasContext? _domContext;
  bool _domConfigured = false;
  int _domWidth = 0;
  int _domHeight = 0;
  ui.Rect? _lastLogicalArea;

  /// The canvas texture format frames are rendered in.
  String get format => device.preferredCanvasFormat;

  /// Whether an active on-screen DOM `<canvas>` is mounted and opted in for
  /// direct WebGPU presentation.
  bool get canPresentToDomCanvas => _resolveDomCanvas() != null;

  void _resetDomCanvas() {
    if (_domConfigured || _domContext != null) {
      try {
        _domContext?.unconfigure();
      } catch (_) {}
    }
    _domCanvas = null;
    _domContext = null;
    _domConfigured = false;
    _domWidth = 0;
    _domHeight = 0;
    _lastLogicalArea = null;
  }

  web.HTMLCanvasElement? _resolveDomCanvas() {
    final element = web.document.getElementById(kDirectPresentCanvasId);
    if (element == null || !element.isA<web.HTMLCanvasElement>()) {
      _resetDomCanvas();
      return null;
    }
    final canvas = element as web.HTMLCanvasElement;
    // Read `data-direct-present` via `getAttribute` (`String?`) rather than
    // `dataset['directPresent']`, whose `package:web` indexer is typed as
    // non-nullable `String` and traps on `undefined` under dart2wasm when the
    // attribute is absent.
    if (canvas.getAttribute('data-direct-present') != '1') {
      _resetDomCanvas();
      return null;
    }
    final previous = _domCanvas;
    if (previous == null ||
        _domContext == null ||
        !canvas.strictEquals(previous).toDart) {
      if (_domConfigured || _domContext != null) {
        try {
          _domContext?.unconfigure();
        } catch (_) {}
      }
      JSObject? rawContext;
      try {
        rawContext = canvas.getContext('webgpu');
      } catch (_) {
        rawContext = null;
      }
      if (rawContext == null) {
        _domCanvas = null;
        _domContext = null;
        _domConfigured = false;
        _domWidth = 0;
        _domHeight = 0;
        _lastLogicalArea = null;
        return null;
      }
      _domCanvas = canvas;
      _domContext = rawContext as GPUCanvasContext;
      _domConfigured = false;
      _domWidth = 0;
      _domHeight = 0;
      _lastLogicalArea = null;
    }
    return canvas;
  }

  void _resize(int width, int height) {
    if (_configured && _canvas.width == width && _canvas.height == height) {
      return;
    }
    _canvas.width = width;
    _canvas.height = height;
    // Premultiplied, matching the linear-premultiplied contract the resolve
    // pass hands to Flutter.
    _context.configure(
      GPUCanvasConfiguration(
        device: device.device,
        format: format,
        alphaMode: 'premultiplied',
        viewFormats: ['$format-srgb'.toJS].toJS,
      ),
    );
    _configured = true;
  }

  /// Clears a [width] by [height] frame to the given color and returns it as
  /// a `ui.Image` without awaiting anything.
  ui.Image clearToImageSync(
    int width,
    int height, {
    required double r,
    required double g,
    required double b,
    required double a,
  }) {
    _resize(width, height);
    final target = _context.getCurrentTexture();
    final encoder = device.device.createCommandEncoder();
    encoder
        .beginRenderPass(
          GPURenderPassDescriptor(
            colorAttachments: [
              GPURenderPassColorAttachment(
                view: target.createView(),
                clearValue: GPUColorDict(r: r, g: g, b: b, a: a),
                loadOp: 'clear',
                storeOp: 'store',
              ),
            ].toJS,
          ),
        )
        .end();
    device.device.queue.submit([encoder.finish()].toJS);
    return _transferSync();
  }

  static const String _blitSource = r'''
@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return textureLoad(source, vec2i(position.xy), 0);
}
''';

  late final GPUBindGroupLayout _blitLayout = device.device
      .createBindGroupLayout(
        _o({
          'entries': [
            {
              'binding': 0,
              'visibility': GPUShaderStage.fragment,
              // Accepts every float format, filterable or not.
              'texture': {'sampleType': 'unfilterable-float'},
            },
          ],
        }),
      );

  late final GPUShaderModule _blitModule = device.device.createShaderModule(
    _o({'code': _blitSource, 'label': 'flutter_scene present blit'}),
  );

  final Map<String, GPURenderPipeline> _blitPipelines = {};

  GPURenderPipeline _blitPipeline(String targetFormat) =>
      _blitPipelines[targetFormat] ??= device.device.createRenderPipeline(
        _o({
          'layout': device.device.createPipelineLayout(
            _o({
              'bindGroupLayouts': [_blitLayout],
            }),
          ),
          'vertex': {'module': _blitModule, 'entryPoint': 'vs'},
          'fragment': {
            'module': _blitModule,
            'entryPoint': 'fs',
            'targets': [
              {'format': targetFormat},
            ],
          },
          'primitive': {'topology': 'triangle-list'},
        }),
      );

  void _submitBlit(
    GPUCanvasContext context,
    GPUTextureView source, {
    required bool srgb,
  }) {
    final targetFormat = srgb ? '$format-srgb' : format;
    final target = context.getCurrentTexture().createView(
      _o({'format': targetFormat}),
    );
    final group = device.device.createBindGroup(
      _o({
        'layout': _blitLayout,
        'entries': [
          {'binding': 0, 'resource': source},
        ],
      }),
    );
    final encoder = device.device.createCommandEncoder();
    encoder.beginRenderPass(
        _o({
          'colorAttachments': [
            {
              'view': target,
              'loadOp': 'clear',
              'storeOp': 'store',
              'clearValue': [0, 0, 0, 0],
            },
          ],
        }),
      )
      ..setPipeline(_blitPipeline(targetFormat))
      ..setBindGroup(0, group)
      ..draw(3)
      ..end();
    device.device.queue.submit([encoder.finish()].toJS);
  }

  /// Copies [source], a single-level 2D view of a [width] by [height] color
  /// texture, to a frame and returns it as a `ui.Image` without awaiting
  /// anything.
  ///
  /// The bytes cross unchanged, rows top-down: an sRGB [source] is drawn
  /// through an sRGB view of the canvas so the decode on load is undone on
  /// store, and the canvas format's channel order is the canvas's concern.
  ui.Image blitToImageSync(
    GPUTextureView source,
    int width,
    int height, {
    required bool srgb,
  }) {
    _resize(width, height);
    _submitBlit(_context, source, srgb: srgb);
    return _transferSync();
  }

  /// Blits [source], a single-level 2D view of a [width] by [height] color
  /// texture, directly into the opted-in on-screen DOM `<canvas>` without
  /// transferring an `ImageBitmap` or wrapping a `ui.Image`.
  ///
  /// Returns `false` when no opted-in DOM `<canvas>` is available, dimensions
  /// or [logicalArea] are empty, or swapchain presentation fails transiently
  /// (such as during tab backgrounding or context transitions), allowing the
  /// caller to fall back to [blitToImageSync].
  bool blitToDomCanvasSync(
    GPUTextureView source,
    int width,
    int height, {
    required ui.Rect logicalArea,
    required bool srgb,
  }) {
    final domCanvas = _resolveDomCanvas();
    final domContext = _domContext;
    if (domCanvas == null ||
        domContext == null ||
        width < 1 ||
        height < 1 ||
        !logicalArea.isFinite ||
        logicalArea.isEmpty) {
      return false;
    }
    try {
      if (!_domConfigured ||
          _domWidth != width ||
          _domHeight != height ||
          domCanvas.width != width ||
          domCanvas.height != height) {
        domCanvas.width = width;
        domCanvas.height = height;
        domContext.configure(
          GPUCanvasConfiguration(
            device: device.device,
            format: format,
            alphaMode: 'premultiplied',
            viewFormats: ['$format-srgb'.toJS].toJS,
          ),
        );
        _domConfigured = true;
        _domWidth = width;
        _domHeight = height;
      }
      if (_lastLogicalArea != logicalArea) {
        final style = domCanvas.style;
        style.left = '${logicalArea.left}px';
        style.top = '${logicalArea.top}px';
        style.width = '${logicalArea.width}px';
        style.height = '${logicalArea.height}px';
        _lastLogicalArea = logicalArea;
      }
      _submitBlit(domContext, source, srgb: srgb);
      return true;
    } catch (_) {
      _domConfigured = false;
      return false;
    }
  }

  ui.Image _transferSync() {
    final bitmap = _canvas.transferToImageBitmap();
    final image = ui_web.createImageFromImageBitmap(bitmap as JSAny);
    if (image is ui.Image) return image;
    throw StateError(
      'createImageFromImageBitmap returned a Future; the synchronous frame '
      'path needs a ui.Image on this renderer.',
    );
  }

  static JSObject _o(Map<String, Object?> fields) =>
      fields.jsify()! as JSObject;

  void dispose() {
    if (_configured) _context.unconfigure();
    _resetDomCanvas();
  }
}
