class ExternalTextureStats {
  int captureCount = 0;
  Duration lastCaptureDuration = Duration.zero;
  Duration minCaptureDuration = Duration.zero;
  Duration maxCaptureDuration = Duration.zero;
  Duration totalCaptureDuration = Duration.zero;

  double get averageCaptureMilliseconds => captureCount == 0
      ? 0
      : totalCaptureDuration.inMicroseconds / captureCount / 1000;

  void record(Duration duration) {
    captureCount++;
    lastCaptureDuration = duration;
    totalCaptureDuration += duration;
    if (captureCount == 1 || duration < minCaptureDuration) {
      minCaptureDuration = duration;
    }
    if (captureCount == 1 || duration > maxCaptureDuration) {
      maxCaptureDuration = duration;
    }
  }

  void reset() {
    captureCount = 0;
    lastCaptureDuration = Duration.zero;
    minCaptureDuration = Duration.zero;
    maxCaptureDuration = Duration.zero;
    totalCaptureDuration = Duration.zero;
  }
}
