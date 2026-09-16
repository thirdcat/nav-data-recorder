# 기존 세션을 이용한 한 시간 R&D 검증 — 2026-09-09

작업 구간: **00:36:37–01:35:09 KST, 약 59분**. [실행 계획과 사전등록](../../docs/METRIC_RECONSTRUCTION_PLAN.md#13-한-시간-연구-세션--2026-09-09-0036-kst-착수)을 먼저 작성하고 실험을 단계적으로 확장했다.
이 디렉터리는 개인 세션에서 파생된 **로컬 연구 기록**이다. 원본 사진/센서 파일을 복사하거나 커밋하지 않았다.

## 판정

**센서를 더 강하게 묶으면 metric 정확도가 자동으로 좋아진다는 가설은 지지되지 않았다.**
관측 연결성과 측정 모델의 불확실성을 먼저 다뤄야 한다. IMU의 상대 회전을 사용하는 경로는 확보했다.
새 3DGS 학습이나 photorealism 개선을 입증한 실험은 이번에 수행하지 않았다.

- IMU: 한 세션에서 보정한 camera/device 회전축과 시간 offset 0으로 **ARKit 54개 중 54개** 검사 통과.
  보정에 사용하지 않은 세션은 53개다. 각 세션의 0.2초 상대 회전 pair ≥10,
  중앙값 ≤1°, p95 ≤3° 기준이며, 실제 세션별 p95의 최댓값은 0.694° 미만이다.
- 중앙 24장으로 고정한 Pi3X 캐시 **18개**: 관측량 기준은 9개, training 관측의 완전 연결은 7개.
  RGBD BA의 ARKit 대비 위치 차이는 15/18개에서 증가했고, IMU 회전 고정 arm은 10/18개에서 증가했다.
  고정 arm의 RGB 재투영 오차는 전부 감소했다. 두 지표를 혼동하면 안 된다.
- 원래의 예비 기준을 모두 통과한 것은 3/18개였다. 후속 연결성 진단을 더하면
  `40ebb3`, `5b88f7` **2개 중앙 구간**만 남는다. 원래 3개 판정을 소급해서 덮어쓰지 않았다.
  두 세션의 앞/뒤 구간 4개를 추가 검증하자 전체 기준 통과는 **0/4개**였다.
- 인접 translation prior 5 cm와 inverse-depth 보간을 각각 추가해도 통과 구간은 같았다.
  단일 규제 강도나 depth pixel floor가 실패 전체를 설명하지 못했다.
- MultiCam 3개: wide RGBD 개선이 ultra-wide에 일부 전달됐지만 15% 전체 pair 개선 기준은 충족하지 못했다.
  `2d9844`, `e92fe5`의 바깥쪽 소수 관측은 개선됐다. 전체 FOV나 scene 품질의 증명은 아니다.
- 30 Hz 세션 4개, 같은 시작/끝 사진을 둔 8개 시간 구간씩:
  30 Hz−5 Hz의 유효 대응점 비율 차이 중앙값은 **7.47, 0.00, 4.95, 1.06 percentage point**.
  사전등록한 +10 pp, 3/4 세션 조건은 미충족이다. 일부 어려운 pair에서는 큰 이득도 있었다.
  같은 구간의 예측 노출 운동량은 960 px 환산 **6.54, 3.10, 4.78, 2.79 px**였다.

![포즈 비교](pose_tradeoff.png)

## 실험 조건과 해석 범위

BA는 첫 camera pose와 K를 고정하고, training landmark와 나머지 pose를 최적화했다.
RGB block soft-L1 1.5 px, LiDAR z 0.03+0.01z m, 기존 pose prior 0.5 m/30°,
seed 42 whole-track 20% 평가 분리, 최대 150 function evaluation을 고정했다.
ARK RGB는 960 px, MultiCam wide는 원본 640 px, 별도 UW 검증은 960 px다.
metric 단위는 depth 및 초기 캐시에서 가져오며 별도 scale 변수를 맞추지 않았다.
ARKit 위치 비교는 한 번의 SE(3) 정렬만 사용한다. Sim(3) scale fitting은 없다.

관측량은 평가 track ≥30, frame coverage ≥80%. RGB 재투영 ≥15% 감소,
depth abs-rel 악화 ≤10%, ARKit 위치 차이 ≥20% 감소,
ARKit 초기화 대조군의 첫 pose gauge 내 이동 중앙값 ≤2 cm를 예비 기준으로 사용했다.
관측 연결성은 뒤늦게 발견된 문제이므로 후속 채택 조건으로 따로 표시한다.
완전 연결도 충분한 시차/잘 조건화된 Hessian을 보장하지 않는다.

**여기서 held-out은 BA optimizer가 보지 않은 track을 뜻한다.**
Pi3X 초기 캐시는 같은 RGB 프레임으로 생성됐고 입력 conditioning/model revision의 완전한 기록도 없다.
따라서 계획 §6의 엄격한 전체 파이프라인 novel-view holdout을 충족한 결과로 쓰면 안 된다.
UW 영상은 이번 BA에 쓰지 않았지만 보정과 기존 캐시의 모든 학습 이력까지 감사한 것은 아니다.
ARKit과 CMDeviceMotion은 센서를 공유하는 처리된 추정치라 독립 GT가 아니다.
회전축의 세션 간 검증은 ARKit 모드에서 했으며 MultiCam으로의 적용은 별도의 검증이 필요하다.
LiDAR held-out 차이 역시 센서 일관성이다. 이번에는 줄자/외부 표면 기준으로 절대 정확도를 재검증하지 않았다.
직선에 가까운 trajectory에서 위치 기반 SE(3) fit의 회전은 불안정할 수 있어,
그 회전 오차 대신 별도 상대 IMU 회전 검사를 해석에 사용했다.

## MultiCam과 관측 빈도

![렌즈 간 전달](multicam_transfer.png)

| 세션 | wide 오차 px: 초기 → RGBD | UW 오차 px: 초기 → RGBD | UW pair | 바깥쪽 관측 수 / 오차 px |
| --- | --- | --- | ---: | --- |
| d06152 | 5.179 → 3.317 | 6.392 → 6.577 | 4 | 0 / 평가 불가 |
| 2d9844 | 4.525 → 0.980 | 6.703 → 5.814 | 56 | 51 / 14.328 → 7.588 |
| e92fe5 | 3.441 → 1.955 | 4.646 → 4.246 | 47 | 45 / 9.266 → 6.641 |

UW는 실제 timestamp에 wide pose를 보간하고 고정 rig를 합성했다. 관측 선택은 arm과 무관하게
mutual SIFT ratio 0.75, F RANSAC 1.5 px, pair당 ≥20점을 사용했다.
wide training track에서 12 px 이내의 source 특징점은 제외했다.
바깥쪽은 정규화 image radius >0.6이며 관측 수는 독립 landmark/scene 표본 수가 아니다.

![시간 간격 비교](high_rate_tracking.png)

같은 0.4초 시작/끝 사진에서 5/10/30 Hz LK 경로만 바꿨다.
각 구간의 차이를 구한 후 중앙값을 취했으므로, 두 arm의 개별 중앙값을 뺀 값과 다를 수 있다.
끝점의 ARKit+depth 예측과 ≤3 px인 관측을 전체 유효 source depth 특징점의 비율로 계산했다.
노출 blur 수치는 기존 `motion_blur.streaks`의 centred exposure 모델 예측이며 픽셀에서 추정한 sharpness가 아니다.

## 재현과 산출물

- [summary.json](summary.json), [census.csv](census.csv): 최종 집계.
- [arkit_census/selection.json](arkit_census/selection.json): 18개 캐시, 선택 범위와 입력 해시.
  각 하위 폴더에 `commands.json`, `prior_commands.json`, `interpolation_commands.json`,
  cache, solver report, 로그가 있다. 각 arm의 B0도 남겨 초기화 차이를 확인할 수 있다.
- [arkit_census/diagnostics.json](arkit_census/diagnostics.json): 실제 training camera 연결 성분과
  LiDAR를 쓰지 않는 held-out RGB Sampson 거리. 이 지표는 scale을 검사하지 못한다.
- [replication/selection.json](replication/selection.json), [replication/summary.json](replication/summary.json):
  성공 중앙 구간과 겹치지 않는 첫/마지막 블록의 재검증.
- `imu_calibration.json`, `imu_all_sessions.json`: 고정 C, train/held-out 구분, 전체 세션 검사.
- `ba_*`, `imu_ba_*`, `fixed_imu_*`: 최초 문제 구간, ARKit 대조군, 3개 MultiCam pilot.
  `2994fa`의 지정 실패 window와 census의 중앙 window는 서로 다르다.
- `uw_*`, `high_rate_*`: UW 관측 목록·해시·개별 pair 결과와 기록 빈도 비교.
- [checks/results.json](checks/results.json): 8개 geometry self-test와 fixture reader 결과.
- [environment.json](environment.json), `code/`, `artifact_manifest.json`: 환경, 코드 snapshot, 산출물 해시.
  모든 최적화는 CPU로 실행했다. 157 optimizer arm은 같은 자료의 대조/민감도 조건들을 포함하며
  독립 실험 또는 독립 장면 157개라는 뜻이 아니다.

```bash
python3 eval/run_local_ba_census.py --data-root /home/myeongcheol/nav_data \
  --calibration logs/research_hour_20260909/imu_calibration.json \
  --out /tmp/nav-ba-census-reproduction --jobs 3
python3 logs/research_hour_20260909/summarize.py
```

기존 출력 경로의 덮어쓰기를 막는 CLI이므로 재실행에는 새 출력 경로를 사용한다.

## 다음 행동

1. **관측으로 연결된 구간만 BA 후보로 삼고 미관측 자유도를 명시한다.**
   frame coverage 외에 covisibility 연결, 시차, 깊이/시선 분포를 검사한다.
   IMU/prior로 메운 연결은 실제 visual/depth 관측과 별도로 표시한다.
2. **IMU는 우선 회전 제약으로 사용한다.** 이번 회전 고정은 자유도 진단이며 최종 모델이 아니다.
   raw IMU가 없는 현 상태에서 translation preintegration으로 곧바로 확장하지 않는다.
   노출 중 회전과 센서 시간 모델, depth 잔차의 조건별 불확실성을 먼저 검증한다.
3. **wide와 UW가 함께 유지하는 landmark를 확충한다.** 바깥쪽 개선의 초기 근거는 있으나 표본이 적다.
   렌즈별 camera model과 rig/시간 오차를 분리한 평가를 유지한다.
4. **GS 비교 전에 프레임 단위 split부터 다시 고정한다.** Pi3X conditioning/BA에서 평가 사진을 뺀
   geometry를 새로 만들고, 같은 물리적 평가 camera·렌즈·해상도에서 GS 외관을 비교한다.
   기존 캐시의 track holdout 결과를 그 증명으로 대체하지 않는다.
