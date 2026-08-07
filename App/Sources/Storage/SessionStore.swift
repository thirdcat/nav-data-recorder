import Foundation

/// On-disk layout and lifecycle for recorded sessions.
///
/// Everything lives under `Documents/sessions/<id>/`, which is what
/// `UIFileSharingEnabled` exposes in the Files app — so manual retrieval and
/// automatic upload read the exact same tree, with no copy step between them.
enum SessionStore {

    enum Filename {
        static let manifest = "manifest.json"
        static let location = "location.jsonl"
        static let heading = "heading.jsonl"
        static let motion = "motion.jsonl"
        static let pose = "pose.jsonl"
        static let planes = "planes.jsonl"
        static let depthIndex = "depth.jsonl"
        static let depthData = "depth.bin"
        static let confidenceData = "confidence.bin"
        static let events = "events.jsonl"
        static let video = "video.mov"
        static let frameIndex = "frames.jsonl"
        /// Subdirectory holding the JPEGs in stills mode.
        static let framesDirectory = "frames"
        /// Marker written once every file for a session is closed. Upload only
        /// considers sessions that have it, so a drive still in progress is
        /// never shipped half-written.
        static let complete = ".complete"
        /// Records which files the upload queue has already acknowledged.
        static let uploadState = ".upload.json"
    }

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static var root: URL {
        documents.appendingPathComponent("sessions", isDirectory: true)
    }

    /// e.g. `20260807-014530-a1b2c3`. Sorts chronologically as a string, which
    /// keeps the session list cheap — no need to open manifests to order them.
    static func makeSessionID(date: Date = Date()) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        fmt.timeZone = TimeZone(secondsFromGMT: 0)
        fmt.locale = Locale(identifier: "en_US_POSIX")
        let suffix = String(UUID().uuidString.prefix(6)).lowercased()
        return "\(fmt.string(from: date))-\(suffix)"
    }

    static func directory(for id: String) -> URL {
        root.appendingPathComponent(id, isDirectory: true)
    }

    @discardableResult
    static func createDirectory(for id: String) throws -> URL {
        let dir = directory(for: id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Session IDs, newest first.
    static func listSessionIDs() -> [String] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: root.path) else { return [] }
        return entries
            .filter { !$0.hasPrefix(".") }
            .filter { var d: ObjCBool = false
                      return fm.fileExists(atPath: directory(for: $0).path, isDirectory: &d) && d.boolValue }
            .sorted(by: >)
    }

    static func readManifest(id: String) -> SessionManifest? {
        let url = directory(for: id).appendingPathComponent(Filename.manifest)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SessionManifest.self, from: data)
    }

    static func writeManifest(_ manifest: SessionManifest) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(manifest)
        let url = directory(for: manifest.id).appendingPathComponent(Filename.manifest)
        try data.write(to: url, options: .atomic)
    }

    static func markComplete(id: String) {
        let url = directory(for: id).appendingPathComponent(Filename.complete)
        FileManager.default.createFile(atPath: url.path, contents: Data())
    }

    static func isComplete(id: String) -> Bool {
        FileManager.default.fileExists(
            atPath: directory(for: id).appendingPathComponent(Filename.complete).path)
    }

    /// Payload files in a session, as paths relative to the session directory.
    ///
    /// Recurses, because stills mode puts thousands of JPEGs in `frames/`, and
    /// a flat listing would silently exclude every one of them from upload.
    /// Dotfiles are excluded deliberately: the upload state and completion
    /// marker are this device's bookkeeping, not part of the dataset.
    static func payloadFiles(id: String) -> [String] {
        let dir = directory(for: id)
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: dir,
                                             includingPropertiesForKeys: [.isRegularFileKey],
                                             options: [.skipsHiddenFiles]) else {
            return []
        }
        var results: [String] = []
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            let path = url.path
            guard path.hasPrefix(dir.path) else { continue }
            var relative = String(path.dropFirst(dir.path.count))
            if relative.hasPrefix("/") { relative.removeFirst() }
            guard !relative.isEmpty else { continue }
            results.append(relative)
        }
        return results.sorted()
    }

    static func url(for id: String, relativePath: String) -> URL {
        directory(for: id).appendingPathComponent(relativePath)
    }

    static func totalBytes(id: String) -> UInt64 {
        payloadFiles(id: id).reduce(0) { sum, relative in
            let values = try? url(for: id, relativePath: relative)
                .resourceValues(forKeys: [.fileSizeKey])
            return sum + UInt64(values?.fileSize ?? 0)
        }
    }

    static func delete(id: String) throws {
        try FileManager.default.removeItem(at: directory(for: id))
    }

    /// Bytes still free on the volume holding the sessions directory.
    ///
    /// A full-rate capture can eat tens of GB an hour, so the recorder checks
    /// this rather than discovering the problem when a write fails mid-drive.
    ///
    /// Queried against Documents, not `root`: the sessions directory does not
    /// exist until the first recording, and asking an absent path for its
    /// volume capacity returns nothing — which would read as "disk full" and
    /// refuse to ever start.
    static func availableCapacity() -> Int64 {
        let values = try? documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }
}
