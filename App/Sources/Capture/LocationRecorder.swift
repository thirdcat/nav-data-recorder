import CoreLocation
import Foundation

/// GPS track capture.
///
/// This is the one stream that keeps running with the screen off, so it is also
/// what holds the app alive in the background. Video and depth stop the moment
/// iOS suspends the camera; location and motion do not.
final class LocationRecorder: NSObject, CLLocationManagerDelegate {

    private let manager = CLLocationManager()
    private let queue: DispatchQueue

    /// Called on `queue`.
    var onLocation: ((LocationSample) -> Void)?
    var onHeading: ((HeadingSample) -> Void)?
    var onEvent: ((String, String) -> Void)?

    private var anchor: Double = 0
    private var running = false

    private(set) var lastLocation: CLLocation?

    init(queue: DispatchQueue) {
        self.queue = queue
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.distanceFilter = kCLDistanceFilterNone
        manager.activityType = .automotiveNavigation
        // iOS will otherwise decide the vehicle has stopped and silently cut
        // updates — which shows up in the data as an unexplained gap.
        manager.pausesLocationUpdatesAutomatically = false
    }

    var authorizationStatus: CLAuthorizationStatus {
        manager.authorizationStatus
    }

    func requestAuthorization() {
        switch manager.authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse:
            // Only meaningful to ask for Always once When-In-Use is granted.
            manager.requestAlwaysAuthorization()
        default:
            break
        }
    }

    func start(anchor: Double) {
        self.anchor = anchor
        guard !running else { return }
        running = true

        let status = manager.authorizationStatus
        if status == .authorizedAlways {
            // Only legal to set with Always authorization; setting it otherwise
            // throws an exception rather than returning an error.
            manager.allowsBackgroundLocationUpdates = true
        } else {
            onEvent?("location.foregroundOnly",
                     "authorization=\(Self.describe(status)); recording stops if the app is backgrounded")
        }

        manager.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            manager.startUpdatingHeading()
        }
    }

    func stop() {
        guard running else { return }
        running = false
        manager.stopUpdatingLocation()
        manager.stopUpdatingHeading()
        manager.allowsBackgroundLocationUpdates = false
    }

    // MARK: - CLLocationManagerDelegate

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let anchor = self.anchor
        queue.async { [weak self] in
            guard let self = self else { return }
            for loc in locations {
                self.lastLocation = loc
                let sample = LocationSample(
                    t: Clock.monotonic(fromWallClock: loc.timestamp, anchor: anchor),
                    wall: loc.timestamp.timeIntervalSince1970,
                    lat: loc.coordinate.latitude,
                    lon: loc.coordinate.longitude,
                    alt: loc.altitude,
                    altEllipsoidal: loc.ellipsoidalAltitude,
                    hAcc: loc.horizontalAccuracy,
                    vAcc: loc.verticalAccuracy,
                    speed: loc.speed,
                    speedAcc: loc.speedAccuracy,
                    course: loc.course,
                    courseAcc: loc.courseAccuracy)
                self.onLocation?(sample)
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let anchor = self.anchor
        queue.async { [weak self] in
            guard let self = self else { return }
            self.onHeading?(HeadingSample(
                t: Clock.monotonic(fromWallClock: newHeading.timestamp, anchor: anchor),
                trueHeading: newHeading.trueHeading,
                magneticHeading: newHeading.magneticHeading,
                accuracy: newHeading.headingAccuracy))
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        queue.async { [weak self] in
            self?.onEvent?("location.error", error.localizedDescription)
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        queue.async { [weak self] in
            guard let self = self else { return }
            self.onEvent?("location.authorization", Self.describe(status))
            // Promote to background capture if the user upgrades to Always
            // partway through a drive.
            if self.running, status == .authorizedAlways {
                manager.allowsBackgroundLocationUpdates = true
            }
        }
    }

    static func describe(_ status: CLAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "notDetermined"
        case .restricted: return "restricted"
        case .denied: return "denied"
        case .authorizedAlways: return "always"
        case .authorizedWhenInUse: return "whenInUse"
        @unknown default: return "unknown"
        }
    }
}
