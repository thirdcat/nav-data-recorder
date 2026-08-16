import SwiftUI

/// Drives `MultiCamRecorder`: a walk recorded through the ultra-wide lens with
/// LiDAR depth beside it, and no poses at all.
///
/// Kept off the main record screen deliberately. The ordinary recorder is an
/// `ARSession` and this is an `AVCaptureMultiCamSession`; the two cannot run
/// together, and a control that silently swapped which one was recording would
/// make it impossible to tell later what a session actually is. Two screens,
/// two session kinds, and `manifest.json` names which.
struct MultiCamRecordView: View {
    @State private var recorder: MultiCamRecorder?
    @State private var running = false
    @State private var status = ""
    @State private var finished: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button(running ? "Stop" : "Start recording") {
                    running ? stop() : start()
                }
                .buttonStyle(.borderedProminent)
                .tint(running ? .red : .accentColor)

                Text("""
                     Walk the room as you would for an ordinary recording. \
                     Texture matters more than usual here — bookshelves, \
                     posters, patterned floor. A blank wall gives the offline \
                     pose model nothing to work with, and there is no tracker \
                     in this mode to fall back on.
                     """)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                Text("""
                     This session has **no poses**. ARKit offers only \
                     wide-angle formats, so reaching the ultra-wide means \
                     leaving the tracker behind; poses are recovered offline \
                     from the images, and metric scale from the depth recorded \
                     alongside them.
                     """)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                if !status.isEmpty {
                    Text(status).font(.callout)
                }
                if let finished {
                    Text(finished)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
        .navigationTitle("Ultra-wide walk")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { if running { stop() } }
    }

    private func start() {
        let made = MultiCamRecorder()
        made.onStatus = { status = $0 }
        made.onFinished = { result in
            running = false
            switch result {
            case .success(let url):
                finished = "wrote \(url.lastPathComponent)"
                status = "done — it will upload with the other sessions"
            case .failure(let error):
                status = "failed: \(error.localizedDescription)"
            }
        }
        recorder = made
        running = true
        finished = nil
        made.start()
    }

    private func stop() {
        status = "finishing…"
        recorder?.stop()
    }
}
