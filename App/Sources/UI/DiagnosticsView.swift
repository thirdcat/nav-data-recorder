import SwiftUI
import UIKit

/// Shows what the hardware actually supports, for questions the documentation
/// cannot answer — chiefly whether an alternative capture path (multi-camera,
/// ultra-wide, LiDAR depth outside ARKit) is available on this device.
struct DiagnosticsView: View {
    @State private var report: String = ""
    @State private var copied = false

    var body: some View {
        ScrollView {
            Text(report.isEmpty ? "Gathering…" : report)
                .font(.system(.caption, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .padding()
        }
        .navigationTitle("Capabilities")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    UIPasteboard.general.string = report
                    copied = true
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                }
            }
        }
        .task {
            // Enumerating every format on every camera takes a beat, so keep it
            // off the main thread and out of app launch.
            let generated = await Task.detached(priority: .userInitiated) {
                CaptureCapabilities.report()
            }.value
            report = generated
        }
    }
}
