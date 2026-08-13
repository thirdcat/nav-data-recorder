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
    @State private var status = "Point the phone at something 0.5–3 m away with a "
        + "few real surfaces in frame, then run. Two cameras open for about five seconds."
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
