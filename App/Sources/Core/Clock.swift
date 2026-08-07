import Foundation
import QuartzCore

/// The single time base every recorder writes into.
///
/// Four subsystems report time in three different domains:
///
/// - `CMDeviceMotion.timestamp` — seconds since boot
/// - `ARFrame.timestamp`        — seconds since boot (`CACurrentMediaTime()`)
/// - `CMSampleBuffer` PTS       — host time, same domain as the two above
/// - `CLLocation.timestamp`     — a wall-clock `Date`
///
/// So the monotonic since-boot clock is the natural master: three of the four
/// already speak it, and it does not jump when the system clock is corrected
/// mid-drive. Every sample is stamped with `t` in that domain; the session
/// manifest carries one `clockAnchor` so `t` can be turned back into Unix time
/// offline (`unix = t + clockAnchor`).
enum Clock {

    /// Current time in the master (monotonic, since-boot) domain.
    static func now() -> Double {
        CACurrentMediaTime()
    }

    /// Offset that maps the master clock onto Unix time: `unix = t + anchor`.
    ///
    /// Sampled once per session at start. It drifts only as far as the system
    /// clock itself is adjusted, which is what we want — every `t` stays
    /// internally consistent even if NTP steps the wall clock mid-recording.
    static func wallClockAnchor() -> Double {
        Date().timeIntervalSince1970 - CACurrentMediaTime()
    }

    /// Converts a `CLLocation`-style wall-clock date into the master domain.
    static func monotonic(fromWallClock date: Date, anchor: Double) -> Double {
        date.timeIntervalSince1970 - anchor
    }
}
