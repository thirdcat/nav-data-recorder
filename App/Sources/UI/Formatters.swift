import Foundation

enum Format {

    static func duration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "--:--" }
        let total = Int(seconds)
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        return h > 0
            ? String(format: "%d:%02d:%02d", h, m, s)
            : String(format: "%02d:%02d", m, s)
    }

    static func bytes(_ count: UInt64) -> String {
        bytes(Int64(clamping: count))
    }

    static func bytes(_ count: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        return formatter.string(fromByteCount: count)
    }

    /// Sessions are named `yyyyMMdd-HHmmss-<suffix>` in UTC. Shown back in the
    /// local time zone, because "when did I record this" is a local question.
    static func sessionTitle(_ id: String) -> String {
        let parts = id.split(separator: "-")
        guard parts.count >= 2 else { return id }
        let input = DateFormatter()
        input.dateFormat = "yyyyMMdd-HHmmss"
        input.timeZone = TimeZone(secondsFromGMT: 0)
        input.locale = Locale(identifier: "en_US_POSIX")
        guard let date = input.date(from: "\(parts[0])-\(parts[1])") else { return id }

        let output = DateFormatter()
        output.dateStyle = .medium
        output.timeStyle = .short
        return output.string(from: date)
    }

    static func coordinate(_ value: Double) -> String {
        String(format: "%.6f", value)
    }

    static func speedKPH(_ metresPerSecond: Double) -> String {
        guard metresPerSecond >= 0 else { return "—" }
        return String(format: "%.0f km/h", metresPerSecond * 3.6)
    }
}
