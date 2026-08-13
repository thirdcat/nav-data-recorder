import Foundation

/// Where uploaded sessions go. Kept separate from `CaptureConfig` because it is
/// device-level configuration, not a property of any one recording.
struct UploadSettings: Codable, Equatable {
    var enabled: Bool = false
    /// Files are PUT to `<baseURL>/<sessionID>/<filename>`.
    var baseURL: String = ""
    /// Sent as `Authorization: Bearer <token>` when non-empty.
    var authToken: String = ""
    /// On by default: a full-rate session is tens of gigabytes and would eat a
    /// mobile plan in one drive.
    var wifiOnly: Bool = true
    /// Off by default. Deleting on the strength of a 200 means one server-side
    /// mistake loses the data permanently; the manual Files-app copy is the
    /// backstop, so the default is to keep both.
    var deleteAfterUpload: Bool = false

    static let `default` = UploadSettings()

    var resolvedBaseURL: URL? {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let url = URL(string: trimmed), url.scheme != nil else { return nil }
        return url
    }

    var isUsable: Bool { enabled && resolvedBaseURL != nil }
}

/// Thin `UserDefaults` wrapper. Small enough that a store abstraction would be
/// more machinery than it saves.
final class AppSettings {
    static let shared = AppSettings()

    private let defaults = UserDefaults.standard
    private enum Key {
        static let captureConfig = "captureConfig"
        static let uploadSettings = "uploadSettings"
    }

    private init() {}

    var captureConfig: CaptureConfig {
        get {
            guard var stored: CaptureConfig = Self.decode(defaults.data(forKey: Key.captureConfig)) else {
                return .default
            }
            // Depth used to be pinned to the stills rate and had no control of
            // its own, so a stored 5 Hz is a leftover rather than a choice. A
            // new default alone would never reach an existing install, and the
            // rate is the whole point of the change — depth registration needs
            // every frame it can get.
            if stored.depthHz < 10 {
                stored.depthHz = CaptureConfig.default.depthHz
            }
            // Same reasoning: confidence was off by default and had no
            // advocate, so a stored `false` is an old default rather than a
            // decision. Depth that cannot be validated is not worth its bytes.
            if !stored.recordConfidence {
                stored.recordConfidence = true
            }
            return stored
        }
        set { defaults.set(Self.encode(newValue), forKey: Key.captureConfig) }
    }

    var uploadSettings: UploadSettings {
        get { Self.decode(defaults.data(forKey: Key.uploadSettings)) ?? .default }
        set { defaults.set(Self.encode(newValue), forKey: Key.uploadSettings) }
    }

    private static func decode<T: Decodable>(_ data: Data?) -> T? {
        guard let data = data else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    private static func encode<T: Encodable>(_ value: T) -> Data? {
        try? JSONEncoder().encode(value)
    }
}
