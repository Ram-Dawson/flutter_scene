import 'dart:async';

import 'package:flutter/foundation.dart' show visibleForTesting;

/// Serializes synchronous CPU transcodes and mipmap builds on Flutter Web and
/// yields a macrotask (`Future.delayed(Duration.zero)`) before starting the
/// next work item once cumulative CPU time in the current slice reaches
/// [sliceBudget].
///
/// On Web (`kIsWeb`), `compute` runs on the main UI thread without yielding a
/// macrotask, so a batch of textures transcoded or mipmapped back-to-back
/// would starve browser rendering and loading progress updates.
final class WebTranscodeTimeSlicer {
  WebTranscodeTimeSlicer({
    this.sliceBudget = const Duration(milliseconds: 8),
    Stopwatch? stopwatch,
    Future<void> Function()? yieldMacrotask,
  }) : _stopwatch = stopwatch ?? Stopwatch(),
       _yieldMacrotask = yieldMacrotask ?? _defaultYieldMacrotask;

  /// Cumulative synchronous CPU budget before yielding a macrotask.
  final Duration sliceBudget;

  final Stopwatch _stopwatch;
  final Future<void> Function() _yieldMacrotask;

  Future<void> _tail = Future<void>.value();

  static Future<void> _defaultYieldMacrotask() =>
      Future<void>.delayed(Duration.zero);

  /// Resets accumulated slice time.
  @visibleForTesting
  void reset() {
    _stopwatch
      ..stop()
      ..reset();
  }

  /// Runs [work] exclusively in FIFO order, yielding a macrotask and resetting
  /// the slice stopwatch beforehand when prior tasks in the current slice have
  /// accumulated at least [sliceBudget].
  Future<T> run<T>(T Function() work) {
    final previous = _tail;
    final gate = Completer<void>();
    _tail = gate.future;
    return _runQueued(previous, gate, work);
  }

  Future<T> _runQueued<T>(
    Future<void> previous,
    Completer<void> gate,
    T Function() work,
  ) async {
    await previous;
    try {
      if (_stopwatch.elapsed >= sliceBudget) {
        _stopwatch.reset();
        await _yieldMacrotask();
      }
      _stopwatch.start();
      try {
        return work();
      } finally {
        _stopwatch.stop();
      }
    } finally {
      gate.complete();
    }
  }
}

/// Shared main-thread CPU time-slicer used on `kIsWeb` for texture transcodes
/// and async mipmap generation.
final WebTranscodeTimeSlicer webTranscodeTimeSlicer = WebTranscodeTimeSlicer();
