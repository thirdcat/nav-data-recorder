# Setup

This project is built and shipped without a Mac. The `.xcodeproj` is generated
by XcodeGen on a cloud macOS runner, signed from an App Store Connect API key,
and delivered to the phone through TestFlight. Xcode is never opened by hand.

The cost of that is the development loop: **nothing is compiled until CI runs
it.** A typo costs a full build cycle, so expect the first few builds to fail on
small things.

## What you need

| Thing | Required? | Notes |
| --- | --- | --- |
| Apple Developer Program | **Yes**, $99/year | TestFlight needs it. Enrolment can take 1–2 days — start it first. |
| Codemagic account | Yes (free tier is enough) | 500 build-minutes/month; a build here is ~10 minutes. |
| iPhone with LiDAR | Yes | iPhone 12 Pro or later. Tested target is the 16 Pro. |
| Charger or power bank | In practice, yes | The screen must stay on for the whole recording (see below). |
| A Mac | **No** | That is the point of this setup. |

## One-time setup

### 1. Enrol in the Apple Developer Program

<https://developer.apple.com/programs/>. Do this first; everything else waits on
it.

### 2. Register the app in App Store Connect

App Store Connect → Apps → **+** → New App.

- Platform: iOS
- Bundle ID: `com.thirdcat.navdatarecorder` — must match `project.yml` and
  `codemagic.yaml`. Register it under Certificates, Identifiers & Profiles →
  Identifiers first if it is not in the dropdown.
- SKU: anything, e.g. `navdatarecorder`

### 3. Create an App Store Connect API key

Users and Access → Integrations → App Store Connect API → **+**

- Access: **App Manager**
- Download the `.p8` file. **It can only be downloaded once.**
- Note the **Issuer ID** and the **Key ID**.

> The `.p8` is a signing credential. It must never be committed — `.gitignore`
> blocks `*.p8`, but the real rule is that it only ever lives in Codemagic's
> encrypted environment.

### 4. Connect Codemagic

1. Sign up at <https://codemagic.io> with the GitHub account that owns this repo
   and give it access to `thirdcat/nav-data-recorder`.
2. Teams → Integrations → App Store Connect → **Add key**.
   - Name it **`NavDataRecorderKey`** — `codemagic.yaml` refers to it by that
     exact name.
   - Paste the Issuer ID, Key ID, and the contents of the `.p8`.
3. Codemagic will pick up `codemagic.yaml` automatically. Signing certificates
   and provisioning profiles are created and rotated from the API key; there is
   nothing to generate by hand.

### 5. Install TestFlight on the iPhone

App Store → TestFlight. Builds appear there a few minutes after Apple finishes
processing them.

## The development loop

```
push to the branch
  └─ Codemagic: brew install xcodegen → xcodegen generate → build → sign → upload
       └─ Apple processes the build (5–15 min)
            └─ TestFlight notification on the phone → install
```

When a build fails, the compiler output is in the Codemagic build log; the tail
of `/tmp/xcodebuild_logs/*.log` is also kept as an artifact.

## First run on the device

Grant, in order:

1. **Camera** — prompted by ARKit on the first recording.
2. **Motion & Fitness** — prompted by CoreMotion.
3. **Location** — say *While Using* at the prompt, then go to
   Settings → NavRecorder → Location → **Always**. iOS will not offer Always in
   the first prompt; it has to be raised afterwards.

Without Always, the GPS track stops the moment the screen locks. Indoors that
matters less than it sounds — GPS is context, not the pose source — but the
recording is cheaper to interpret with it present.

## Things the hardware imposes

- **The camera cannot record in the background.** iOS suspends capture when the
  app is not frontmost. IMU and GPS continue (that is what the `location`
  background mode buys), but video, depth and pose stop. The app keeps the
  screen awake while recording for exactly this reason — keep it on a charger.
- **It will get hot.** ARKit plus LiDAR plus HEVC encoding is close to a
  worst-case thermal load. With *Pause camera when hot* enabled (the default),
  video and depth pause at `serious` thermal state and resume when it drops
  back. Every transition is written to `events.jsonl`.
- **Storage runs out fast.** Roughly 100 MB per minute at the defaults; the
  Settings tab shows a live estimate. Recording stops automatically at 2 GB free.
- **Tracking will drop on blank walls.** ARKit needs visual texture. Move slowly,
  keep furniture and edges in frame, and expect `limited:insufficientFeatures`
  events in featureless corridors.

## Getting data off the phone

Both routes are available, and they read the same files:

- **Files app** — On My iPhone → NavRecorder → `sessions/`. Copy a session
  folder to iCloud Drive or anywhere else. No server needed.
- **Automatic upload** — Settings tab → Upload. Each file is `PUT` to
  `<base URL>/<session id>/<filename>` with an optional
  `Authorization: Bearer <token>`. Transfers use a background `URLSession`, so
  they continue after the app is suspended and survive a relaunch. Wi-Fi only is
  on by default.

  *Delete after upload* is **off** by default, and worth leaving off until the
  server side is known good — a 200 response is not proof the bytes landed
  somewhere you can still read next month.

## Reading a session

```bash
python3 tools/read_session.py /path/to/20260807-014530-a1b2c3
python3 tools/read_session.py <dir> --align
```

To exercise the reader without a phone:

```bash
python3 tools/make_test_session.py /tmp/fixture
python3 tools/read_session.py /tmp/fixture/20260807-014530-fixture
```

The format itself is in [DATA_FORMAT.md](DATA_FORMAT.md).
