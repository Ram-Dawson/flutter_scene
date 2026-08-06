import 'package:flutter_test/flutter_test.dart';
import 'package:example_app/external_texture_stats.dart';

void main() {
  test('tracks count, min, max, last, and average capture duration', () {
    final stats = ExternalTextureStats();

    stats.record(const Duration(milliseconds: 4));
    stats.record(const Duration(milliseconds: 11));

    expect(stats.captureCount, 2);
    expect(stats.lastCaptureDuration, const Duration(milliseconds: 11));
    expect(stats.minCaptureDuration, const Duration(milliseconds: 4));
    expect(stats.maxCaptureDuration, const Duration(milliseconds: 11));
    expect(stats.averageCaptureMilliseconds, 7.5);
  });

  test('reset clears all measurements', () {
    final stats = ExternalTextureStats()
      ..record(const Duration(milliseconds: 4));

    stats.reset();

    expect(stats.captureCount, 0);
    expect(stats.lastCaptureDuration, Duration.zero);
    expect(stats.averageCaptureMilliseconds, 0);
  });
}
