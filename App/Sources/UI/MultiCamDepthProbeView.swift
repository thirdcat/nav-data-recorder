import SwiftUI
import UIKit

/// Runs `MultiCamDepthProbe` and shows its report, in the same read-and-paste
/// shape as the capabilities screen.
///
/// This one has to be driven by a button rather than starting on appear: it
/// opens two cameras at once, which is a real power and thermal cost, and a
/// screen that did that merely by being navigated to would be a bad neighbour
/// to a device already fighting its thermal budget.
struct MultiCamDepthProbeView: View {
    @State private var probe: MultiCamDepthProbe?
    // The scene matters as much as the code here. A frame filled by one close
    // wall is the easiest case the LiDAR ever sees, and it makes the hole-
    // filling comparison look harmless. Ask for depth in the frame.
    @State private var status = "Aim across a room, not at a nearby wall — take in "
        + "something close AND something 3 m or more away, ideally including glass or a "
        + "dark surface. Two cameras open for about eight seconds."
    @State private var report = ""
    @State private var running = false
    @State private var copied = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button(running ? "Running…" : "Run probe") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(running)

                Text(status)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                if !report.isEmpty {
                    Text(report)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
        .navigationTitle("Multi-cam depth")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    UIPasteboard.general.string = report
                    copied = true
                } label: {
                    Label(copied ? "Copied" : "Copy",
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                }
                .disabled(report.isEmpty)
            }
        }
        .onDisappear {
            probe?.cancel()
            probe = nil
        }
    }

    private func start() {
        report = ""
        copied = false
        running = true
        status = "starting…"

        let probe = MultiCamDepthProbe()
        probe.onStatus = { status = $0 }
        probe.onFinished = { result in
            running = false
            switch result {
            case .success(let text):
                report = text
                status = "done — copy the report"
            case .failure(let error):
                report = ""
                status = "failed: \(error.localizedDescription)"
            }
        }
        self.probe = probe
        probe.run()
    }
}
