import Foundation

/// Decides when the next sample of a stream is due, when that stream is being
/// sampled down from a faster source.
///
/// The obvious formulation — "accept once at least half an interval has passed
/// since the last accepted sample" — is wrong, and wrong in a way that is easy
/// to miss: against a 60 Hz source with a 5 Hz target, the first frame past
/// 100 ms always qualifies, so the stream runs at 10 Hz. Twice the configured
/// rate, twice the storage.
///
/// This advances a fixed schedule instead. The long-run rate matches the
/// setting, and a stall in the source resynchronises rather than producing a
/// burst of catch-up samples.
struct RateGate {
    private let interval: Double
    private var nextDue: Double?

    init(hz: Double) {
        self.interval = 1.0 / max(0.01, hz)
    }

    /// Whether a sample taken at `t` (master clock seconds) should be kept.
    mutating func shouldFire(at t: Double) -> Bool {
        guard let due = nextDue else {
            nextDue = t + interval
            return true
        }
        guard t >= due else { return false }
        // On schedule: keep the cadence. Far behind: restart from now, so a gap
        // in the source does not fire several times in a row trying to catch up.
        nextDue = (t - due < interval) ? due + interval : t + interval
        return true
    }
}
