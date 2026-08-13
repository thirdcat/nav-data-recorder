import CoreGraphics
import SwiftUI

/// The depth map as a picture, with the numbers that say whether it is worth
/// anything.
///
/// The screen has only ever reported one number about depth — the fraction of
/// the map ARKit rates medium or high — and that number is silent about both of
/// the ways this sensor fails indoors. Glass, mirrors and dark surfaces return
/// nothing at all, so a frame can be 78% confident and still be a hole exactly
/// where the geometry mattered, and no percentage can say *where* the hole is.
/// Returns past about 5 m are not usable, so facing down a long corridor
/// produces a frame that is almost entirely confident and almost entirely out
/// of range. Both are obvious at a glance in a picture.
///
/// The image is drawn as it arrives — the camera's native landscape
/// orientation, never rotated — so this belongs in a landscape layout at around
/// 160x120 points, where 256x192 lands at a little under 2x on a Retina screen.
struct DepthPreview: View {
    let image: CGImage?
    let p10: Double
    let median: Double
    let p95: Double
    let invalid: Double

    /// Half the frame returning nothing. Deliberately a loose "most of this is
    /// missing" rule and not a calibrated threshold — the picture is what says
    /// how bad it is, and this only decides when to make someone look at it.
    private static let missingLimit = 0.5

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.primary.opacity(0.05))
                if let image = image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        // Nearest neighbour. This is drawn larger than it is,
                        // and smoothing would blend a one-pixel dropout into
                        // its neighbours — the dropouts are the whole point.
                        .interpolation(.none)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    Text("Depth appears while recording")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(6)
                }
            }
            // Pinned to the depth map's own 4:3 whatever width the layout
            // gives it, so the placeholder occupies exactly the space the image
            // will and nothing moves when depth starts arriving.
            .aspectRatio(4.0 / 3.0, contentMode: .fit)

            Text(caption)
                .font(.caption2.monospacedDigit())
                .foregroundStyle(image != nil && invalid >= Self.missingLimit
                                 ? .orange : .secondary)
                .lineLimit(1)
                // At 160 points this line does not fit at the larger text
                // sizes, and a statistic truncated mid-number is worse than a
                // small one.
                .minimumScaleFactor(0.6)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spokenLabel)
    }

    /// Three distances, then the holes. The distances are what the picture
    /// cannot quantify — a p95 past 5 puts most of the frame beyond where the
    /// sensor is trustworthy, which looks the same on screen as a dark room —
    /// and the last number is how much of it returned nothing at all.
    private var caption: String {
        guard image != nil else { return "—" }
        guard median > 0 else { return "no returns · \(percent(invalid)) empty" }
        return "\(metres(p10))/\(metres(median))/\(metres(p95)) m"
            + " · \(percent(invalid)) no return"
    }

    /// The caption spelled out, since three numbers separated by slashes are
    /// legible only next to the picture they describe.
    private var spokenLabel: String {
        guard image != nil else { return "Depth preview. No depth yet." }
        guard median > 0 else { return "Depth preview. No returns at all." }
        return "Depth preview. Median \(metres(median)) metres, most returns"
            + " between \(metres(p10)) and \(metres(p95))."
            + " \(percent(invalid)) of the frame has no return."
    }

    private func metres(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private func percent(_ value: Double) -> String {
        "\(Int((max(0, min(1, value)) * 100).rounded()))%"
    }
}
