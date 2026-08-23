import CoreGraphics
import SwiftUI

/// Drives `MultiCamRecorder`: a walk recorded through the ultra-wide lens with
/// LiDAR depth beside it, and no poses at all.
///
/// Kept off the main record screen deliberately. The ordinary recorder is an
/// `ARSession` and this is an `AVCaptureMultiCamSession`; the two cannot run
/// together, and a control that silently swapped which one was recording would
/// make it impossible to tell later what a session actually is. Two screens,
/// two session kinds, and `manifest.json` names which.
///
/// **The layout is `RecordView`'s**, for `RecordView`'s reason: the app is
/// landscape only, so there is no room to scroll past anything, and a preview
/// you have to scroll to is not a preview. It used to be a `ScrollView` with a
/// button and three paragraphs in it and nothing to aim — which is how a
/// checkerboard sequence came back with the board past 0.70 of the image radius
/// in 3 frames of 221.
struct MultiCamRecordView: View {
    @State private var recorder: MultiCamRecorder?
    @State private var running = false
    @State private var status = ""
    @State private var finished: String?
    @State private var preview: CGImage?
    @State private var lenses: MultiCamRecorder.LensGeometry?

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            UltraWideAimingPreview(image: preview,
                                   geometry: lenses,
                                   isRecording: running)
            sidebar
                .frame(width: 300)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .navigationTitle("Ultra-wide walk")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { if running { stop() } }
    }

    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Button(running ? "Stop" : "Start recording") {
                    running ? stop() : start()
                }
                .buttonStyle(.borderedProminent)
                .tint(running ? .red : .accentColor)

                legend

                // The two paragraphs about what this recorder is are reference
                // material, read once before starting. While it is running the
                // sidebar is beside a live picture that has to be aimed, and a
                // wall of text next to it is what the last capture had.
                if !running {
                    Text("""
                         Walk the room as you would for an ordinary recording. \
                         Texture matters more than usual here — bookshelves, \
                         posters, patterned floor. A blank wall gives the \
                         offline pose model nothing to work with, and there is \
                         no tracker in this mode to fall back on.
                         """)
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text("""
                         This session has **no poses**. ARKit offers only \
                         wide-angle formats, so reaching the ultra-wide means \
                         leaving the tracker behind; poses are recovered \
                         offline from the images, and metric scale from the \
                         depth recorded alongside them.
                         """)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                // Shown before the button is pressed, because it is the one
                // thing about this screen that is now set somewhere else.
                Text(presetSummary)
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
        }
    }

    /// What the two overlays mean, in the fewest words that make them
    /// actionable. The percentages are computed from the session's own formats
    /// rather than typed in, so this line cannot end up describing a box that
    /// is no longer being drawn.
    ///
    /// Each string is one literal, `+` nowhere in it: `Text` only reads
    /// markdown out of a `LocalizedStringKey`, and a concatenation picks the
    /// plain-`String` overload instead, which would put the asterisks on the
    /// screen.
    private var legend: some View {
        let footprint = UltraWideAiming.footprint(for: lenses)
        let width = Int((footprint.widthFraction * 100).rounded())
        let height = Int((footprint.heightFraction * 100).rounded())
        return VStack(alignment: .leading, spacing: 5) {
            Text("""
                 **Yellow box** — the wide lens, and so the LiDAR depth: the \
                 middle \(width) % x \(height) % of this frame. Outside it \
                 there is picture and no depth.
                 """)
            Text("""
                 **Nine zones** — park the board in each and hold it still. \
                 The four corners are the shot every capture so far has \
                 missed, and the one a lens model is short of.
                 """)
            if !footprint.exact {
                Text("""
                     The ultra-wide is not running the format the box was \
                     measured on, so the box is approximate — the session's \
                     notes carry the field of view it did run.
                     """)
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    /// What the Settings preset means on *this* path, in the preset's own
    /// numbers rather than the ARKit recorder's — they are deliberately not the
    /// same, and a screen that implied they were would be the inheritance this
    /// change is about.
    private var presetSummary: String {
        let preset = AppSettings.shared.captureConfig.activePreset ?? .vln
        var text = "Preset: \(preset.rawValue) — "
            + "\(Int(preset.multiCamStillsHz)) Hz per lens"
        if let cap = preset.maxDurationSeconds {
            text += ", stops at \(Int(cap)) s"
        }
        if let delay = preset.lockExposureAfterSeconds {
            text += ", exposure locked after \(Int(delay)) s"
        }
        return text + "."
    }

    private func start() {
        // Read straight from the store rather than through the environment:
        // `RecordingCoordinator.config` writes through to `AppSettings` on
        // every change, so this is the same value, and it does not make this
        // screen depend on an object it otherwise has no use for.
        //
        // `activePreset` is nil for a hand-edited config, and this path then
        // falls back to `.vln` — the 5 Hz behaviour it has always had. A screen
        // that inherited "some custom rate" from the other recorder is the
        // failure this whole change exists to stop.
        let preset = AppSettings.shared.captureConfig.activePreset ?? .vln
        let made = MultiCamRecorder(preset: preset)
        made.onStatus = { status = $0 }
        made.onLensGeometry = { lenses = $0 }
        // Guarded on `running` because the capture queue is still draining when
        // `stop()` returns: without it a frame in flight repaints the preview
        // after the session has ended, and a still picture of a finished walk
        // sitting under a live overlay is the one thing this screen must not
        // show.
        made.onPreview = { image in
            if running { preview = image }
        }
        made.onFinished = { result in
            running = false
            preview = nil
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
        preview = nil
        lenses = nil
        made.start()
    }

    private func stop() {
        status = "finishing…"
        recorder?.stop()
    }
}
