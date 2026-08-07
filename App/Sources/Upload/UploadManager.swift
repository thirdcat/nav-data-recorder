import Foundation
import UIKit

/// Ships completed sessions to a server over a background `URLSession`.
///
/// Background rather than foreground because the transfers are enormous — a
/// full-rate hour is tens of gigabytes — and the user is not going to sit
/// watching a progress bar. iOS keeps these going after the app is suspended
/// and relaunches it when they finish.
///
/// Uploading never deletes anything by default. The Files-app copy stays put
/// unless the user explicitly opts into deletion, so a server-side mistake
/// cannot silently destroy a drive.
final class UploadManager: NSObject, ObservableObject, URLSessionDelegate, URLSessionTaskDelegate {

    static let sessionIdentifier = "com.thirdcat.navdatarecorder.upload"

    /// A background `URLSession` may only have one owner per identifier, and
    /// iOS can relaunch the app straight into the delegate before any SwiftUI
    /// view exists — so this has to be reachable without a view hierarchy.
    static let shared = UploadManager()

    struct Progress: Equatable {
        var pendingFiles = 0
        var inFlightFiles = 0
        var failedFiles = 0
        var lastError: String?
        var lastCompleted: String?
    }

    @Published private(set) var progress = Progress()
    @Published var settings: UploadSettings {
        didSet {
            AppSettings.shared.uploadSettings = settings
            let updated = settings
            stateQueue.async { [weak self] in self?.currentSettings = updated }
            if settings.isUsable { resumePending() }
        }
    }

    /// `settings` is bound to the UI and mutated on main, but the upload path
    /// runs on `stateQueue` and on URLSession's delegate queue. This is the copy
    /// those paths read, so no background code touches the published property.
    private var currentSettings: UploadSettings

    /// Set by the app delegate when iOS relaunches us to report finished
    /// background transfers; called once the session has drained its events.
    var backgroundCompletionHandler: (() -> Void)?

    private var urlSession: URLSession!
    private let stateQueue = DispatchQueue(label: "nav.upload.state")

    override init() {
        let stored = AppSettings.shared.uploadSettings
        self.settings = stored
        self.currentSettings = stored
        super.init()

        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.allowsCellularAccess = !stored.wifiOnly
        // Lets iOS pick a moment that suits power and network conditions rather
        // than starting a 40 GB transfer the instant a drive ends.
        config.isDiscretionary = true
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        self.urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    // MARK: - Public API

    /// Queues every not-yet-uploaded file in a session. Safe to call repeatedly;
    /// files already in flight or already acknowledged are skipped.
    func enqueue(sessionID: String) {
        guard settings.isUsable else { return }
        guard SessionStore.isComplete(id: sessionID) else { return }

        urlSession.getAllTasks { [weak self] tasks in
            guard let self = self else { return }
            let inFlight = Set(tasks.compactMap { $0.taskDescription })
            self.stateQueue.async {
                let uploaded = Self.uploadedFiles(sessionID: sessionID)
                var queued = 0
                for url in SessionStore.payloadFiles(id: sessionID) {
                    let name = url.lastPathComponent
                    let key = "\(sessionID)/\(name)"
                    guard !uploaded.contains(name), !inFlight.contains(key) else { continue }
                    if self.startUpload(sessionID: sessionID, file: url, key: key) {
                        queued += 1
                    }
                }
                let total = queued + inFlight.count
                DispatchQueue.main.async {
                    self.progress.inFlightFiles = total
                }
            }
        }
    }

    /// Re-queues every complete session that still has unsent files. Called at
    /// launch and whenever the upload settings change, so a session recorded
    /// before a server was configured is not stranded.
    func resumePending() {
        guard settings.isUsable else { return }
        for id in SessionStore.listSessionIDs() where SessionStore.isComplete(id: id) {
            enqueue(sessionID: id)
        }
    }

    /// Files in this session that the server has acknowledged.
    static func uploadedFiles(sessionID: String) -> Set<String> {
        let url = SessionStore.directory(for: sessionID)
            .appendingPathComponent(SessionStore.Filename.uploadState)
        guard let data = try? Data(contentsOf: url),
              let names = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return Set(names)
    }

    /// Whether every payload file in a session has been acknowledged.
    static func isFullyUploaded(sessionID: String) -> Bool {
        let payload = Set(SessionStore.payloadFiles(id: sessionID).map { $0.lastPathComponent })
        guard !payload.isEmpty else { return false }
        return payload.isSubset(of: uploadedFiles(sessionID: sessionID))
    }

    // MARK: - Internals

    /// Returns true if a task was created. Runs on `stateQueue`.
    private func startUpload(sessionID: String, file: URL, key: String) -> Bool {
        let settings = currentSettings
        guard let base = settings.resolvedBaseURL else { return false }

        let target = base
            .appendingPathComponent(sessionID, isDirectory: true)
            .appendingPathComponent(file.lastPathComponent)

        var request = URLRequest(url: target)
        request.httpMethod = "PUT"
        request.setValue(Self.contentType(for: file), forHTTPHeaderField: "Content-Type")
        let token = settings.authToken.trimmingCharacters(in: .whitespacesAndNewlines)
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // A background session can only upload from a file, never from Data.
        let task = urlSession.uploadTask(with: request, fromFile: file)
        task.taskDescription = key
        task.resume()
        return true
    }

    private static func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "json": return "application/json"
        case "jsonl": return "application/x-ndjson"
        case "mov": return "video/quicktime"
        default: return "application/octet-stream"
        }
    }

    /// Records that the server has taken this file. Runs on `stateQueue`.
    private func markUploaded(sessionID: String, filename: String) {
        var names = Self.uploadedFiles(sessionID: sessionID)
        names.insert(filename)
        let url = SessionStore.directory(for: sessionID)
            .appendingPathComponent(SessionStore.Filename.uploadState)
        if let data = try? JSONEncoder().encode(Array(names).sorted()) {
            try? data.write(to: url, options: .atomic)
        }

        if currentSettings.deleteAfterUpload, Self.isFullyUploaded(sessionID: sessionID) {
            try? SessionStore.delete(id: sessionID)
        }
    }

    // MARK: - URLSessionTaskDelegate

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let key = task.taskDescription ?? ""
        let parts = key.split(separator: "/", maxSplits: 1).map(String.init)
        let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        let success = error == nil && (200...299).contains(status)

        stateQueue.async { [weak self] in
            guard let self = self else { return }
            if success, parts.count == 2 {
                self.markUploaded(sessionID: parts[0], filename: parts[1])
            }
            let message: String?
            if let error = error {
                message = error.localizedDescription
            } else if !success {
                message = "HTTP \(status) for \(key)"
            } else {
                message = nil
            }

            DispatchQueue.main.async {
                self.progress.inFlightFiles = max(0, self.progress.inFlightFiles - 1)
                if success {
                    self.progress.lastCompleted = key
                } else {
                    self.progress.failedFiles += 1
                    self.progress.lastError = message
                }
            }
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}
