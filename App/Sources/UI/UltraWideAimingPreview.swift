import CoreGraphics
import Foundation
import SwiftUI

/// Where the wide camera — and therefore the LiDAR depth — lands inside the
/// ultra-wide frame, and the nine places a calibration board has to be parked.
///
/// Everything here is a **fraction of the ultra-wide frame**, so it survives a
/// change of capture resolution untouched. It does not survive a change of
/// *field of view*, which is exactly why the two angles it needs are read off
/// the session's own active formats rather than written down as constants: see
/// `MultiCamRecorder.LensGeometry`.
///
/// The arithmetic, in full, for the formats this recorder currently pins:
///
/// ```
///   wide lens, active 640x480, videoFieldOfView 69.52744
///     half-angles                       34.7637 x 27.4997 deg
///   ultra-wide, active 3840x2160, paraxial focal 1466.00 px
///     footprint half-extents            1466 * tan(theta)
///                                       = 1018 px of 1920, 763 px of 1080
///                                       = 53 % of the width, 71 % of the height
/// ```
enum UltraWideAiming {

    // MARK: - The two lenses

    /// The wide camera's half-angles for the format `selectDepthFormat`
    /// currently lands on: `videoFieldOfView` 69.52744 at 640x480, halved, with
    /// the vertical derived from that format's own 4:3.
    ///
    /// A fallback only — the live value from the session wins. It is here so a
    /// screen with no session yet still draws the box in the right place, and
    /// so the number a reader of this file is being asked to check is written
    /// down beside the arithmetic that uses it.
    static let defaultWideHalfAngleH: Double = 34.7637
    static let defaultWideHalfAngleV: Double = 27.4997

    /// The pinned ultra-wide format's **paraxial** focal length divided by its
    /// own half-width: 1466.00 px over 1920, from `docs/UW_TRANSFER_PREREG.md`.
    ///
    /// Paraxial, and not `(W/2)/tan(hfov/2)`. That expression evaluates to
    /// 1441.56 px on this format and is the trap that page names and measures:
    /// `videoFieldOfView` is a whole-frame angle that already contains the
    /// distortion, so it belongs inside a tangent and never under a half-width.
    /// The two readings differ by 1.7 %, which is 17 px on a 1018 px box edge —
    /// about the width of the line drawn here, so the choice does not visibly
    /// move the box. It is made the right way round anyway, because the same
    /// mistake in a rectifier cost two attempts before it was named.
    ///
    /// Divided by the half-width rather than kept in pixels so it is
    /// dimensionless: a 1920x1080 readout of the same field of view gives the
    /// same box.
    static let focalPerHalfWidth: Double = 1466.0 / 1920.0

    /// The ultra-wide field of view `focalPerHalfWidth` was measured on, and
    /// how far the session may drift from it before the box is called
    /// approximate.
    static let calibratedFieldOfView: Double = 106.2007
    static let fieldOfViewTolerance: Double = 0.5

    /// The footprint of the wide camera inside the ultra-wide frame, as
    /// fractions of that frame's full width and height.
    struct Footprint {
        let widthFraction: Double
        let heightFraction: Double
        /// False when the ultra-wide's active field of view is not the one
        /// `focalPerHalfWidth` was measured on, so the paraxial focal length
        /// had to be carried across by the ratio of two tangents. That carry
        /// assumes the lens's distortion at the frame edge is unchanged, which
        /// is an assumption and not a measurement — hence the flag, and the
        /// line the screen puts beside the box when it is set.
        let exact: Bool
    }

    static func footprint(for geometry: MultiCamRecorder.LensGeometry?) -> Footprint {
        var halfAngleH = defaultWideHalfAngleH
        var halfAngleV = defaultWideHalfAngleV
        var ratio = focalPerHalfWidth
        var aspect = 16.0 / 9.0
        var exact = true

        if let lenses = geometry {
            if lenses.wideFieldOfView > 1, lenses.wideWidth > 0, lenses.wideHeight > 0 {
                // The horizontal half-angle straight from the device. The
                // vertical follows from the wide format's own aspect ratio
                // because the two axes share one focal length — every
                // depth-carrying format on the LiDAR device is 4:3, so this is
                // 27.49 today, but it is derived rather than assumed.
                halfAngleH = lenses.wideFieldOfView / 2
                halfAngleV = degrees(atan(tan(radians(halfAngleH))
                                          * lenses.wideHeight / lenses.wideWidth))
            }
            if lenses.ultraWideWidth > 0, lenses.ultraWideHeight > 0 {
                aspect = lenses.ultraWideWidth / lenses.ultraWideHeight
            }
            if lenses.ultraWideFieldOfView > 1,
               abs(lenses.ultraWideFieldOfView - calibratedFieldOfView) > fieldOfViewTolerance {
                ratio = focalPerHalfWidth
                    * tan(radians(calibratedFieldOfView / 2))
                    / tan(radians(lenses.ultraWideFieldOfView / 2))
                exact = false
            }
        }

        // The half-extent in pixels is `f * tan(theta)`; dividing by the frame's
        // half-width turns that into the box's width as a fraction of the whole
        // frame. The vertical is the same expression against the frame's
        // half-*height*, which is why the aspect ratio appears once.
        let width = ratio * tan(radians(halfAngleH))
        let height = ratio * tan(radians(halfAngleV)) * aspect
        return Footprint(widthFraction: min(1, max(0, width)),
                         heightFraction: min(1, max(0, height)),
                         exact: exact)
    }

    // MARK: - The nine zones

    struct Target {
        /// Frame-normalised, centre (0, 0), frame edge +/-1.
        let x: Double
        let y: Double
        let corner: Bool
    }

    /// How far out the eight non-central zones sit, on each axis.
    ///
    /// 0.80 on both axes puts the four corner zones at exactly 0.80 of the
    /// half-diagonal, because scaling both axes by 0.80 scales the diagonal by
    /// 0.80. That is the region last night's capture never reached: the board
    /// got to 0.78 at its best and **3 frames of 221 got past 0.70**, which is
    /// the whole reason a lens model cannot be fitted from it. The four edge
    /// zones land at 0.70 and 0.39 of the half-diagonal on a 16:9 frame — the
    /// horizontal pair is roughly the radius the old capture treated as its
    /// limit, so they read as the easy version of the corners.
    static let targetReach: Double = 0.80

    /// Side of one zone, as a fraction of the frame's short side. Small enough
    /// that all nine together cover under a tenth of the picture.
    static let targetSide: Double = 0.13

    /// Centre, four edge midpoints, four corners. The corners are flagged
    /// because they are the ones that matter and the ones that are hardest to
    /// hold: at 0.80 of the half-diagonal the board is at the edge of a
    /// 106-degree lens, where it is small, steeply foreshortened and moving.
    /// Written with the type name rather than bare `targetReach`: a static
    /// stored property's initialiser referring to another static of the same
    /// type is legal, and spelling it out costs nothing on a build that cannot
    /// be run locally.
    static let targets: [Target] = [
        Target(x: 0, y: 0, corner: false),
        Target(x: -UltraWideAiming.targetReach, y: 0, corner: false),
        Target(x: UltraWideAiming.targetReach, y: 0, corner: false),
        Target(x: 0, y: -UltraWideAiming.targetReach, corner: false),
        Target(x: 0, y: UltraWideAiming.targetReach, corner: false),
        Target(x: -UltraWideAiming.targetReach,
               y: -UltraWideAiming.targetReach, corner: true),
        Target(x: UltraWideAiming.targetReach,
               y: -UltraWideAiming.targetReach, corner: true),
        Target(x: -UltraWideAiming.targetReach,
               y: UltraWideAiming.targetReach, corner: true),
        Target(x: UltraWideAiming.targetReach,
               y: UltraWideAiming.targetReach, corner: true),
    ]

    /// The near end of the depth preview's own colour ramp, (255, 232, 56).
    /// Reused on purpose: the box is the region that has depth behind it, and
    /// the other place on this app's screens where that colour appears is the
    /// depth map itself.
    static let footprintColour = Color(red: 1.0, green: 0.910, blue: 0.220)

    // MARK: - Drawing

    /// Draws both overlays into a rectangle that is exactly the drawn image.
    ///
    /// Immediate mode rather than a stack of `Shape` views: this is twelve
    /// strokes that all derive from one rectangle, and as a view tree it would
    /// be a `ForEach` over nine positions plus a SwiftUI body large enough to be
    /// worth worrying about — `docs/SCAN_PRESET.md` already names a slow-typing
    /// body as one of the things only CI can catch.
    static func draw(_ footprint: Footprint,
                     into context: inout GraphicsContext,
                     size: CGSize) {
        let w = Double(size.width)
        let h = Double(size.height)
        guard w > 1, h > 1 else { return }

        // Zones first, so the footprint box draws over them where they cross.
        let side = min(w, h) * targetSide
        for target in targets {
            let rect = CGRect(x: w * (1 + target.x) / 2 - side / 2,
                              y: h * (1 + target.y) / 2 - side / 2,
                              width: side,
                              height: side)
            let path = Path(roundedRect: rect, cornerRadius: CGFloat(side * 0.22))
            // A dark stroke under a light one, so both survive a white wall and
            // a dark floor. Cheaper and steadier than a shadow.
            context.stroke(path, with: .color(Color.black.opacity(0.30)),
                           lineWidth: target.corner ? 3.5 : 2.5)
            context.stroke(path,
                           with: .color(Color.white.opacity(target.corner ? 0.90 : 0.34)),
                           lineWidth: target.corner ? 1.6 : 1)
        }

        let box = CGRect(x: w * (1 - footprint.widthFraction) / 2,
                         y: h * (1 - footprint.heightFraction) / 2,
                         width: w * footprint.widthFraction,
                         height: h * footprint.heightFraction)
        let boxPath = Path(box)
        context.stroke(boxPath, with: .color(Color.black.opacity(0.45)), lineWidth: 3.5)
        context.stroke(boxPath, with: .color(footprintColour), lineWidth: 1.6)
    }

    private static func radians(_ value: Double) -> Double { value * .pi / 180 }
    private static func degrees(_ value: Double) -> Double { value * 180 / .pi }
}

/// The ultra-wide lens, live, with the two things about this frame that an
/// operator cannot see and cannot recover afterwards.
///
/// **Why this screen has a preview at all.** It did not, and a checkerboard
/// sequence shot through this recorder came back unusable: 221 frames, 24 %
/// detected, and the board past 0.70 of the image radius in three of them.
/// There was nothing on the screen to aim at — the ARKit recorder has had a
/// preview since the beginning and this path never did.
///
/// **The box is the overlay that matters.** The LiDAR depth is the *wide*
/// camera's, and inside a 106-degree ultra-wide frame the wide lens covers the
/// middle 53 % x 71 %. Outside that rectangle there is image and no depth, and
/// that is both the region a lens calibration has to cover and the region every
/// capture so far has missed. Nothing about holding the phone says where the
/// edge is.
struct UltraWideAimingPreview: View {
    let image: CGImage?
    let geometry: MultiCamRecorder.LensGeometry?
    let isRecording: Bool

    /// The same convention as `RecordView.previewOrientation`, and deliberately
    /// not a second one. The app is landscape-only — `Info.plist` lists
    /// `UIInterfaceOrientationLandscapeRight` and nothing else — and this
    /// recorder never sets `videoOrientation` on the ultra-wide connection, so
    /// buffers arrive in the sensor's own landscape. Every frame this recorder
    /// has written to disk is 3840x2160 rather than 2160x3840, which is the
    /// evidence for it rather than an assumption about it.
    private let previewOrientation: Image.Orientation = .up

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black)
            if let image = image {
                // `aspectRatio(.fit)` sizes this view to the letterboxed
                // picture, so the overlay attached to it is in frame
                // coordinates and needs no rect arithmetic of its own.
                Image(decorative: image, scale: 1, orientation: previewOrientation)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .overlay { aiming }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                placeholder
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spokenLabel)
    }

    private var aiming: some View {
        let footprint = UltraWideAiming.footprint(for: geometry)
        return Canvas { context, size in
            UltraWideAiming.draw(footprint, into: &context, size: size)
        }
        .allowsHitTesting(false)
    }

    private var placeholder: some View {
        VStack(spacing: 6) {
            Image(systemName: "camera.metering.none")
                .font(.largeTitle)
            Text(isRecording
                 ? "Waiting for frames…"
                 : "The ultra-wide appears while recording")
                .font(.caption)
                .multilineTextAlignment(.center)
        }
        .foregroundStyle(.white.opacity(0.6))
        .padding()
    }

    private var spokenLabel: String {
        guard image != nil else {
            return isRecording ? "Ultra-wide preview. Waiting for frames."
                               : "Ultra-wide preview. Appears while recording."
        }
        let footprint = UltraWideAiming.footprint(for: geometry)
        let width = Int((footprint.widthFraction * 100).rounded())
        let height = Int((footprint.heightFraction * 100).rounded())
        return "Ultra-wide preview. The wide lens and the depth cover the "
            + "middle \(width) by \(height) per cent. Nine target zones, four "
            + "of them in the corners."
    }
}
