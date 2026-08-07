import AVFoundation
import SwiftUI

/// Drives `UltraWideProbe`: shoot a handful of ultra-wide frames with their
/// calibration, so `tools/rectify_ultrawide.py` has something real to work on.
///
/// Kept out of the Record tab on purpose. This stops the ARKit session and
/// produces no poses; it is a measurement, not a recording, and mixing the two
/// would invite shooting a session that turns out to have no trajectory.
struct UltraWideProbeView: View {
    @StateObject private var model = ProbeModel()

    var body: some View {
        List {
            Section {
                Text("""
                     Shoot a scene with long straight edges — a doorframe, a skirting board, the join between wall and ceiling. Straightness is what the rectification is checked against, so a scene with nothing straight in it cannot answer the question.

                     Move between shots. Stand about two metres back so the wide edges of the frame have something in them.
                     """)
                .font(.footnote)
                .foregroundStyle(.secondary)
            }

            Section("Capture") {
                if model.running {
                    Button {
                        model.shoot()
                    } label: {
                        Label("Take a shot  (\(model.shots))", systemImage: "camera")
                    }
                    Button("Done") { model.finish() }
                } else {
                    Button {
                        model.start()
                    } label: {
                        Label("Start", systemImage: "play.circle")
                    }
                }
            }

            if !model.log.isEmpty {
                Section("Log") {
                    ForEach(Array(model.log.enumerated().reversed()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }

            if let out = model.finished {
                Section("Written to") {
                    Text(out).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    Text("Pull it off the phone the same way as a session, then run tools/rectify_ultrawide.py on the ultra-wide directory.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .navigationTitle("Ultra-wide probe")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { model.cancel() }
    }
}

private final class ProbeModel: ObservableObject {
    @Published var running = false
    @Published var shots = 0
    @Published var log: [String] = []
    @Published var finished: String?

    private var probe: UltraWideProbe?

    func start() {
        let probe = UltraWideProbe()
        probe.onStatus = { [weak self] line in self?.append(line) }
        probe.onFinished = { [weak self] result in
            guard let self = self else { return }
            self.running = false
            switch result {
            case .success(let url):
                // The last two components are enough to find it; the full
                // container path is noise on a phone screen.
                self.finished = url.pathComponents.suffix(2).joined(separator: "/")
                self.append("finished")
            case .failure(let error):
                self.append("failed: \(error.localizedDescription)")
            }
        }
        self.probe = probe
        probe.start { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                self.running = true
                self.shots = 0
                self.finished = nil
                self.append("started")
            case .failure(let error):
                self.append("could not start: \(error.localizedDescription)")
            }
        }
    }

    func shoot() {
        probe?.capture()
        shots += 1
    }

    func finish() {
        probe?.finish()
    }

    /// Leaving the screen saves rather than discards. Every shot is already on
    /// disk by then, so throwing away the calibration that describes them would
    /// be the one destructive thing this screen could do.
    func cancel() {
        probe?.finish()
        probe = nil
        running = false
    }

    private func append(_ line: String) {
        log.append(line)
        if log.count > 60 { log.removeFirst(log.count - 60) }
    }
}
