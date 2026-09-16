# Metric scale과 photorealistic 3DGS 재구성 실행 계획

작성: **2026-09-08** / 최종 갱신: **2026-09-16**

이 문서는 LiDAR·wide/ultra-wide 카메라·IMU·Pi3X를 이용한 재구성 연구의
**현재 계획, 진행 상태, 판단 근거를 관리하는 기준 문서**다.
기존 실험의 상세 기록은 [POSE.md](POSE.md), [3DGS.md](3DGS.md),
각 사전등록 문서에 남긴다. 새 결과로 방향을 바꾸면 이 문서의 현재 계획을
갱신하고, 마지막 변경 이력에 이전 판단과 변경 이유를 남긴다.

**프로젝트 목적:** 멀티 카메라로 다양한 시점·시야·해상도·촬영 조건의 유효한 2D 영상을
최대한 확보하고, LiDAR depth·IMU·visual feature 기반 SLAM과 Pi3X를 융합해 각 영상의
metric pose와 장면 기하를 정확히 추정한다. 이를 이용해 metric scale을 유지하면서
미사용 평가 시점의 PSNR과 시각적 품질이 높은 3DGS를 만들고, 같은 좌표계에 시뮬레이터가
받는 collision 기하를 함께 만든다. BA·IMU 대조는 이 전체 경로의 일부를 검증하는 실험이다.

**최종 용도 (2026-09-16 확인):** 이 앱은 real2sim2real 파이프라인의 **real2sim 절반을
맡는 스캐너**다. 산출물은 로봇 시뮬레이션 환경 — 한 metric 좌표계에 정렬된 mesh(충돌·물리)와
3DGS(외관)이고, VLN 에피소드와 3DGS 품질은 그 중간 목표였다. 같은 최종 목표를 가진
[LiteReality-Agent](https://github.com/LiteReality)(로컬 `~/uv_workspace/LiteReality-Agent`)는
RoomPlan 껍질 + 생성 자산 + MuJoCo export 로 **물리·의미 층**을 이미 갖고 있고, 우리는
**관측 충실도 층**(LiDAR·다중뷰 기하, metric 검증, 3DGS)을 갖고 있다. 두 저장소는 같은
입력(iPhone LiDAR + ARKit 포즈, 1920×1440 + 256×192, 같은 좌표 규약)에서 출발하므로
경쟁이 아니라 한 파이프라인의 두 절반이다. 그래서 collision 기하(R7)는 보조 목표가 아니라
제3축이고, "두 층이 우리 캡처 위에서 붙는가"가 지금 가장 큰 미지수다 — §1.6.

## 1. 현재 상태와 다음 작업

**R3-O1/F1/F2/G1 완료: 관측 민감도 22구간, 평가 사진을 제외한 Pi3X 6세션, 실제 3DGS 4개 학습을 검증했다. 기존 BA는 기본 재구성 경로로 채택하지 않는다.**
R2-P1의 전역 정렬은 채택하지 않는다. 회전 기반 순차 연결의 이득은 관찰됐지만,
전역화의 추가 이득이 없었고 별도 영상 검증의 표본도 부족했다.
R1의 과거 MultiCam export/split 복원과 R2의 독립 검증은 남아 있다. 앞선 한 시간에는
CPU 국소 BA 대조·민감도 조건 157개를 실행했다. 이번 후속에서는 사진 단위 holdout을
Pi3X 입력부터 적용하고 동일 초기 점군·평가 camera의 GS 포즈 대조까지 완료했다.
데이터만으로 시작 가능한 실험과 추가 촬영이 필요한 판정은 §1.1에 구분했다.

핵심 판단: 각 Pi3X window에 metric scale을 부여하는 경로와 LiDAR 감독을
사용하는 3DGS 학습 경로는 있다. 포즈 연구의 중심은 **관측을 공유하는
전역·국소 최적화로 구간 내부 변형까지 줄이고, 영상 형성의 불일치를 분리하는 것**이며
그 품질 개선은 아직 가설이다. **2026-09-16 부터 그 앞에 sim-ready 기하 축(R7 → S1~S5)이
선다** — 최종 산출물이 시뮬레이터 환경이고, 그 층은 ARKit 포즈로 BA 없이 시작할 수 있으며,
같은 목표의 LiteReality-Agent 가 그 층을 이미 갖고 있어 우리 캡처 위에서 붙는지가
가장 싸게 답할 수 있는 가장 큰 미지수이기 때문이다. 순서는 §1.6 표.

| ID | 작업 | 상태 | 완료를 판단할 산출물 |
| --- | --- | --- | --- |
| R0 | 기존 성과와 해석 한계 검토 | 완료 | §3의 근거와 재해석 |
| R1 | 기준 데이터·실행 조건·평가 프로토콜 고정 | 진행 — 새 6세션 frame holdout·2세션 GS 재현 완료, 과거 MultiCam 복원 필요 | 재현한 baseline, 고정 split, 평가·실행 명세 |
| R2 | 모든 window의 위치·회전을 사용하는 전역 정렬 | 진행 — P1 완료·미채택, 독립 관측 검증 필요 | W0/W1 비교, 스케일·드리프트·정합 결과 |
| R3 | RGB＋LiDAR＋rig의 metric bundle adjustment | 진행 — 관측 민감도·사진 holdout·4개 GS 학습 완료, 공동 개선 불충족 | W1/W2 비교, 독립 검증과 고정 recipe의 3DGS 결과 |
| R4 | IMU 회전 제약 추가 | 진행 — 54세션 회전 검사 완료, BA 효과는 구간 의존 | W2/W3 비교, 유효한 조건과 실패 조건 |
| R5 | appearance·노출·rolling shutter 모델 검증 | 진행 — 30 Hz 4세션 관측량·노출 운동 진단 완료 | 고정된 geometry 조건에서의 영상 모델 비교 |
| R6 | 반복 촬영·다른 장면·장거리로 검증 확대 | 예정 | metric·정합·외관을 함께 만족하는 적용 범위 |
| R7 | 동일 metric 좌표계의 sim-ready 기하 (제3축, 2026-09-16 승격) | 예정 — S1~S5 로 구체화. BA 결과와 독립이며 ARKit 포즈로 시작 | 시뮬레이터 로드 + 안정성 + collision/support 게이트 (S3) |
| S1 | 우리 세션 → LiteReality 캡처 포맷 변환기 | 예정 | 변환 세션이 저쪽 reader·point cloud 빌더를 통과, 포즈·K 왕복 오차 0 |
| S2 | RoomPlan 과 `ARRecorder` 의 `ARSession` 공유 측정 | 예정 — 촬영 동작만 바꾸고 계측기는 안 건드림 | 같은 걸음 on/off 2 arm: 포맷 유지, 드롭 0, 포즈 차·depth 신뢰도·열 |
| S3 | LiteReality 파이프라인을 우리 캡처 1개에 끝까지 | 대기 — S1, S2 | MuJoCo 로드, 5 s shake 드리프트, 검출/실제 오브젝트 수, fallback 수 |
| S4 | R7 산출물을 sim-ready 계약으로 정의 | 대기 — S3 결과 | §2 표의 제3축 판정 기준 확정 |
| S5 | 3DGS ↔ RoomPlan 껍질 정렬 검증, 하이브리드 씬 조립 | 대기 — S2, S3 | 벽 평면 대 splat 잔차; 저쪽 body + 우리 splat 이 한 좌표계 |
| V1 | 온디바이스 블러 미터·휘도 (외생 신호) | 예정 — 오프라인 검증 먼저 | 54 세션 inventory 대비 신호가 결과를 추적, 그 뒤 한 줄 HUD |
| V2 | 커버리지 BEV (range → 표면별 방향 수) · 정지 직후 요약 | 예정 | 42 세션 커버리지 표 대비 일치; triage 규칙이 폰에서 같은 판정 |
| V3 | 브라우저 뷰어 골격 (궤적·프러스텀·클라우드·render\|real) | 예정 | `trajectories.png` 류를 대체 |
| C1 | 선명한 정지 촬영·보정·독립 치수 기준 확보 | 부분 확보 — 줄자·보드 있음, 장면 전체 독립 검증 세트 필요 | §7의 촬영 세트와 외부 기준 |

**다음 작업은 §1.6 의 우선순위 표를 따른다.** 아래 항목은 그 표의 R3~R5 행이 유지하는
연구 방향이다.

- training track의 camera 연결성, 시차와 깊이/시선 분포를 확인하고, 실제 관측으로
  연결되지 않은 부분의 위치를 BA가 임의로 확정하지 않게 한다. IMU나 Pi3X prior는
  부족한 관측을 보완하는 사전정보로 표시하며 독립 증거로 세지 않는다.
- 검증한 camera/device 회전축을 사용하되, IMU hard rotation은 진단 조건으로 남긴다.
  depth·RGB 잔차와 시간/노출 조건별 불확실성을 검사한 뒤 상대 회전 제약을 설계한다.
- wide/UW에 공통으로 유지되는 landmark, 특히 LiDAR cone 바깥의 평가 관측을 늘린다.
  현재 wide 내부 개선은 크지만 전체 UW로의 전달은 작고 바깥쪽 표본은 적다.
- 이번에 평가 사진을 Pi3X·BA에서 제외한 새 geometry와 GS 대조를 확보했다.
  새 시간 블록에서 Pi3X, 회전만 교체, RGBD BA, soft IMU BA를 비교해 회전 고정과
  translation 최적화의 영향을 분리한다. 기존 개발 세트의 결과를 확증 근거로 재사용하지 않는다.
- 30 Hz는 motion이 큰 pair의 연결을 돕는 후보로 유지하되 선명한 GT로 간주하지 않는다.
  기존 줄자/보드는 독립 scale·보정 검증에 쓰고, C1은 장면 전체 치수와 sharp view의
  부족분만 보충한다. 관측량·센서 일관성의 개선을 절대 정확도/photorealism으로 바꾸어 말하지 않는다.

상태는 `예정 / 진행 / 완료 / 대기 / 보류`로 표시한다. `대기`에는 필요한
입력이나 외부 작업을 적고, `보류`에는 근거와 재개 조건을 적는다.
실험이 끝났다는 것과 가설이 지지됐다는 것은 별도로 기록한다.

### 1.1 확보 데이터로 검증 가능한 범위 — 2026-09-09 감사

**결론: 새 촬영 없이 R2, R3, IMU 회전 제약, 외관 모델의 상대 개선을
검증할 수 있다. 현재 데이터만으로 장면 전체의 절대 정확도와 photorealism을
보장할 수 있다는 뜻은 아니다.** 데이터 부족과 알고리즘·평가 구현 미완료를 구분한다.

#### 실제 파일과 기준 세션

원본 루트는 `/home/myeongcheol/nav_data`다. manifest가 있는 **80개 세션**을
읽었으며 ARKit 54개, MultiCam 26개다. 모두 depth·motion 기록이 있고,
MultiCam 중 22개는 두 렌즈 영상, 17개는 세션의 활성 depth intrinsics가 있다.
인덱스에 적힌 사진 12,854장에 대해 존재·기록 크기를 대조했고,
depth/confidence binary의 인덱스 끝 위치와 파일 크기도 확인했다.
이 범위에서 누락·잘림·manifest 수량 불일치는 없었다.
전체 사진 디코딩, 모든 depth 값의 품질, 보정 정확성을 검증한 것은 아니다.

아래 사진 수는 원본 인덱스 기준이며, 학습에 쓸 수 있는 수와 다르다.
시간은 manifest의 녹화 길이가 아닌 **첫 사진부터 마지막 사진까지**다.

| 세션 ID | 확보한 관측 | 우선 용도 |
| --- | --- | --- |
| `20260813-162849-2994fa` | ARKit RGB 172장/34.2 s, normal 포즈 166장, 13×24 Pi3X window 포즈 캐시 | 크게 실패한 연결의 R2 대조군 |
| `20260813-162946-7d3d52` | ARKit RGB 90장/17.8 s, normal 83장, 6×24 window 캐시 | 상대적으로 양호한 연결을 망가뜨리지 않는지 확인 |
| `20260820-211848-d06152` | UW 290장＋wide 277장/57.8 s, depth 1,376개, motion 5,791개, 활성 depth K | R3 비동시 두 렌즈＋LiDAR 융합의 주 개발 세션 |
| `20260818-160918-2d9844` | UW 58장＋wide 63장/13.3 s, 활성 depth K, 두 렌즈 추정 궤적 | 짧은 MultiCam 회귀 대조군 |
| `20260809-074458-cb4586` | ARKit RGB 172장/34.2 s, normal 167장, 기존 146 train/21 eval과 checkpoint | 기존 3DGS recipe와 포즈 영향의 기준 |
| `20260820-211951-87bc2c` | ARKit RGB 263장/52.4 s, normal 257장, 기존 평가 렌더 33장 | 같은 공간의 다른 경로, blur가 큰 외관 대조군 |
| `20260823-001336-6db873` | RGB·depth 각각 1,747개, 약 30 Hz/58.2 s, 포즈 약 60 Hz | 영상 주기를 바꾸는 통제 및 노출 중 운동 모델의 예비 실험 |
| `20260823-222350-7d9994` | RGB·depth 각각 533개, 약 30 Hz/17.7 s, 약 60 Hz 포즈, 3 s 뒤 노출 잠금 기록 | 노출을 고정한 ARKit에서 R5 실험 |
| `20260823-222432-f48c4e` | UW 204장＋wide 163장/20.3 s, 활성 depth K, 두 렌즈 노출 잠금 | 새 MultiCam 검증 후보; wide 잠금 시 AE 조정 중이었다는 기록 있음 |
| `20260823-224237-8c0134`, `20260823-224301-8eaccd` | 각각 UW/wide 101/80장, 98/79장; 약 10 s, 활성 depth K와 노출 잠금 | 짧은 추가 MultiCam 검증 후보 |
| `20260814-201543-505b2c`, `20260814-201702-e11854`, `20260814-201754-e9d8f7` | 알려진 줄자 길이 1 m, 2 m, 2 m; RGB 102/215/126장 | 외부 길이 기준의 국소 metric 검사 |
| `20260823-003245-1e87b0`부터 `20260823-003324-899c75`까지 보드 6개 take | 두 렌즈＋depth; 마지막 정지 take는 각 13장/2.4 s | 카메라 모델과 운동 중 비투영 변형의 대조 |
| `20260823-053048-b94b35` | normal 101장, 기존 88 train/13 eval, checkpoint | 근거리 coverage가 좋은 기존 외관 기준 |

`6db873`/`7d9994`/`f48c4e`의 중간 사진도 확인했다. 실제 실내 장면이며
흐린 영상이 포함된다. 고속 기록이나 노출 잠금 자체를 선명한 정답 영상으로 취급하지 않는다.
다른 30 Hz ARKit `d4fa51`/`2b38b6`도 있으나 9.9/6.9 s로 짧아 보조군으로 둔다.
위 신규 후보들의 서로 같은 장면 여부·충분한 시차·재방문·사용 가능한 선명한 뷰 수는
아직 전수 검증하지 않았다. 세션 수를 서로 독립적인 장면 수로 세지 않는다.

#### 지금 수행한 작은 검증

**A. 캐시만으로 기존 연결을 CPU에서 재현했다.**
`pi3win/*.npz`에는 `frames`, `pred`, `scale`만 있다.
타임스탬프·참조 포즈는 대응하는 원본/`pi3traj`와 연결할 수 있지만,
Pi3X의 local depth·confidence와 당시 conditioning 인자·모델 revision은 캐시에 없다.
이 캐시는 연결 알고리즘 비교에는 충분하고, LiDAR 측정 수준의 재최적화에는 부족하다.

| 세션 | 기존 free 연결 | seam scale=1인 위치 기반 연결 | 해석 |
| --- | --- | --- | --- |
| `2994fa` | ARKit 대비 평균 위치 차이 2.452 m | 2.463 m | scale 고정만으로 실패가 해소되지 않음 |
| `7d3d52` | 0.261 m | 0.244 m | 큰 실패 사례만 개선하고 양호한 사례를 망가뜨리는지 검사할 대조군 |

각 궤적에 **전역 SE(3) 하나만** 맞추고 위치 차이의 평균을 계산했다.
이 값은 RMSE가 아니며 ARKit도 절대 ground truth는 아니다.
기존 저장 궤적과 재생성한 free 궤적의 최대 위치 차이는 각각 3.3 µm 미만,
0.8 µm 미만이다. `2994fa`의 문제 overlap은 위치로 정렬한 뒤에도 회전 추정이
중앙값 약 28.3° 어긋난다. 회전 관측·이상 window 처리를 함께 시험할 이유가 있다.
**새 W1을 구현하거나 W1이 개선한다고 확인한 실험은 아니다.**

**B. 저장 RGB 렌더를 직접 재채점했다.**
원래 저장된 해상도에서 각 PNG 쌍의 PSNR을 계산했다.

| arm | 평가 사진 수 | 프레임별 PSNR의 중앙값 |
| --- | --- | --- |
| `d06152` free / depth / median | 각 71 | 16.508 / 16.759 / 16.495 dB |
| `d06152` depth wide-only | 35 | 17.655 dB |
| `d06152` depth mixed의 wide 부분 | 동일한 35 | 16.941 dB |
| `87bc2c` ARKit | 33 | 22.676 dB |

wide-only/mixed의 35개 GT PNG는 파일 해시까지 같았다.
**mixed−wide-only의 짝지은 차이**는 평균 −0.764 dB, 중앙값 −0.705 dB이며,
mixed가 이긴 사진은 15/35다. 두 렌즈 융합 개선을 검증할 실제 실패 기준이 남아 있다.
프레임의 시간적 상관 때문에 이 승패 수만으로 통계적 유의성을 주장하지 않는다.
문서의 중앙값과 DN evaluator의 평균값이 다른 이유도 확인했다.
이번에는 RGB만 재채점했으며, 기존 깊이·SSIM 수치를 새로 재현했다고 표시하지 않는다.

checkpoint는 위 arm들과 `cb4586`, `b94b35`에서 확인했다.
다만 `d06152`와 `87bc2c` config가 가리키는 `/tmp/gs_*` export는 현재 없다.
저장된 GT/pred RGB·raw depth는 있어 재채점할 수 있지만,
새 arm을 학습하기 전에는 **원본 프레임 대응·export·split을 복원하고 일치 여부를 검사**해야 한다.
`cb4586`/`b94b35`의 기존 export와 split은 `gs3d/data/`에 남아 있다.

#### 가설별 가능한 판정과 필요한 준비

| 질문 | 지금 가능한 실험 | 아직 필요한 것 / 판정 한계 |
| --- | --- | --- |
| R2: 전역 window 정렬이 순차 연결의 변형을 줄이는가? | 캐시 두 세션에서 위치＋회전 정렬, 강건한 window 가중, 모든 중복 추정을 사용하는 W1 비교. ARKit 전역 SE(3) 차이와 별도 영상 정합을 함께 측정 | W1 구현과 사전 고정 기준 필요. 캐시에 재방문 제약은 없으므로 영상에서 따로 만들고, 평가용 대응점은 최적화에서 제외 |
| R3: LiDAR의 metric 정보를 UW 주변부까지 전달할 수 있는가? | `d06152`에서 wide-only → 두 렌즈 → 두 렌즈＋depth/rig의 통제. 중앙 depth와 연결된 트랙의 주변부 재투영, 제외한 native depth, 동일 wide 평가 뷰 및 UW 뷰 비교 | window 재추론·캐시, BA 구현, 비동시 궤적 평가 필요. LiDAR FOV 밖 절대 깊이는 트랙 일관성만으로 보증 불가 |
| R4: IMU가 회전·연결 안정성을 보완하는가? | 같은 입력에서 회전 제약 on/off, 각속도가 큰 구간과 작은 구간을 나눠 영상 잔차·회전·깊이 일관성 비교 | 약 100 Hz CMDeviceMotion은 있음. 카메라–IMU 축·시각 검증 필요. raw IMU 사전적분과 독립 관성 scale 검증을 그대로 수행할 데이터는 아님 |
| R5: pose와 blur/appearance 중 어디를 바꾸어야 하는가? | `cb4586`/`87bc2c`의 기존 결과 재사용. 30 Hz ARKit에서는 같은 원본을 5/10/30 Hz로 간격 선택하고 동일 평가 시점을 유지하는 통제 가능. `7d9994`와 MultiCam scan 3개로 노출 잠금 이후 진단 | 고정 geometry에서 영상 모델을 바꾸는 학습 구현 필요. 30 Hz 이웃 프레임은 sharp GT가 아니며, 촘촘한 랜덤 split은 피한다. 촬영 간 scene/조명 변화가 있으므로 단순 scan-vln 비교로 잠금 효과를 단정하지 않음 |
| 절대 metric scale이 맞는가? | 줄자 세션에서 알려진 길이를 최적화에 쓰지 않고, 각 추정 궤적/재구성으로 다시 측정. 기준 길이에 맞춰 사후 rescale하지 않음 | 기존 `tape_scale.py`는 ARKit 포즈 누적 점군을 사용하므로 새 궤적 평가에 맞게 바꿔야 함. 길이 3개는 국소 검사이며 장거리·주변부·장면 전체의 1% 보장은 불가 |
| 여러 촬영을 합쳐 더 좋은 3DGS를 만드는가? | `5bd1ed`/`cb4586`와 기존 merge 기준을 이용해 공통 영역 품질의 퇴행을 검사 | 새 영역의 선명한 독립 평가 뷰와 정확한 표면 기준이 부족. 공통 영역 개선과 추가 coverage는 따로 판정 |

**실험 설계에서 반영할 실제 관측 조건:**

- `d06152`의 저장 wide 사진에서 가장 가까운 UW 사진까지의 차이는 중앙값
  **58.0 ms**, p95 **95.7 ms**다. 반면 UW→nearest depth는 11.6/20.7 ms다.
  저장 주기·위상이 다른 결과이며 시계 offset이나 하드웨어 동기 정확도의 추정치가 아니다.
  두 렌즈의 프레임 번호를 묶어 동일 시각 rigid pair로 취급하지 않는다.
- MultiCam 26개 모두 저장된 LiDAR confidence map이 없다. Pi3X confidence와
  동일시하지 않고, native depth의 유효성·불연속·시간 차를 별도로 반영한다.
- MultiCam 사진 행에는 개별 exposure/ISO가 없다. scan 3개는 manifest notes에
  잠금 시각의 조건이 남아 있고 wide 33.34 ms/UW 16.66 ms다.
  잠금 전 구간을 제외한 예비 실험은 가능하지만, 전 구간의 노출 이력을 복원한 것은 아니다.
  노출 시간이 두 배라는 사실만으로 두 렌즈의 pixel blur도 두 배라고 해석하지 않는다.
- motion은 주 후보들의 사진 시간 범위를 거의 모두 덮지만 MultiCam 시작/끝의
  일부 사진은 범위 밖이다. 실제 timestamp 교집합에서 실험한다.
  일부 ARKit pose 스트림의 중복/역순 timestamp도 정렬·중복 처리 후 사용한다.
- 기존 Pi3X→export 경로는 먼저 영상으로 포즈를 추정한 뒤 GS holdout을 정한다.
  기존 성적은 **포즈 추정에 평가 영상이 사용될 수 있는 GS 단계 평가**로 유지한다.
  처음부터 보지 않은 뷰의 end-to-end 성적을 주장할 새 실험은 Pi3X 이전에 split하고,
  평가 포즈 획득에 사용한 관측도 명시한다.

**실행 순서 조정:** 먼저 두 캐시 세션에서 R2의 가장 작은 반증 실험을 만든다.
양호한 세션의 퇴행과 실패 구간의 독립 영상 정합을 함께 통과하면 `d06152`에 확장한다.
그동안 R1에서 MultiCam export를 복원하고 R3용 관측/split을 고정한다.
R5는 새로 확인한 30 Hz·노출 잠금 자료의 진단부터 시작할 수 있으므로
새 촬영이 끝날 때까지 전부 기다리지 않는다. C1은 최종 절대 정확도·선명도 검증을 보충한다.

로컬 근거와 재실행 스크립트:
[`logs/research_feasibility_20260909/README.md`](../logs/research_feasibility_20260909/README.md).
원본 사진·binary·모델은 복사하거나 커밋하지 않았다.

### 1.2 R2-P1 — 캐시 기반 window 정렬 실험

2026-09-09 착수. 아래 실행 조건과 판정 기준은 새 arm 실행 전에 기록한다.
본 실험은 window 내부 포즈와 LiDAR에서 얻은 scale을 고정하고,
window별 SE(3)와 중복 프레임의 통합만 바꾼다.
깊이·confidence 캐시가 없어 scale의 측정 수준 재최적화와 R3 BA는 포함하지 않는다.

- 입력: §1.1에서 해시를 기록한 `pi3win/2994fa.npz`, `7d3d52.npz`.
  ARKit 참조는 평가에만 사용하며 optimizer에 전달하지 않는다.
- 대조군: **W0** 기존 `depth` 위치 기반 순차 연결/첫 추정 유지;
  **WR** 중복 프레임 회전으로 정렬하는 순차 연결/첫 추정 유지.
- 새 arm: **WG-L** 모든 중복 관측의 위치·회전 잔차를 전역 최소제곱;
  **WG-R** 같은 목적함수에 3차원 잔차 블록별 soft-L1 적용.
  WG 두 arm은 변환된 모든 추정의 위치 평균·회전 평균으로 프레임 포즈를 만든다.
  첫 window의 SE(3)는 identity로 고정한다. 새 scale이나 ARKit prior는 없다.
- 위치 0.05 m, 회전 5°로 잔차를 정규화한다. 이는 이번 pilot의 고정된
  모델링 선택이며 보정된 센서 표준편차라는 주장은 아니다. 실제 결과에 맞춘 sweep은 하지 않는다.
  한 프레임이 여러 window에 있어도 총 pair 가중치는 1로 정규화한다.
- 검증: 정확한 합성 궤적 복원, 순수 회전 overlap, 연결되지 않은 graph의 거부,
  잘못된 입력과 비양수 scale 거부, 이상 관측을 넣은 강건성 통제를 먼저 실행한다.
- 실제 평가: 동일 프레임 전체의 전역 SE(3) 정렬 후 ARKit 평균/RMS 위치 차이,
  회전 차이; window 정렬에 사용하지 않은 영상 대응점의 depth 기반 양방향 재투영.
  대응점은 pose로 선택하지 않고 시간 간격과 영상 매칭으로 고정한다.
  원본 영상·depth는 Pi3X 생성 과정에 사용됐을 수 있으므로 end-to-end 미관측 GT는 아니다.
  영상 pair는 균등한 16개 시작점에서 사진 5/15/50개 간격, 시작/끝 각 3개 사진의
  조합으로 제안한다. 폭 640 px, mutual SIFT ratio 0.75, F-RANSAC 1.5 px;
  양쪽 depth 0.2–5 m, confidence ≥2, 3×3 depth 범위 <0.1 m를 고정한다.
  뒤로 투영되거나 크게 벗어난 점도 제외하지 않고 이미지 대각선 길이로 오차를 제한한다.
- **WG-R 확장 후보 판정:** `2994fa` 평균 위치 차이가 W0보다 ≥10% 감소하고,
  `7d3d52`의 증가는 ≤10%; 영상 재투영의 세션별 pair 중앙값 증가는 ≤5%.
  영상 검증은 세션마다 유효한 pair ≥10개, pair당 depth가 있는 대응 ≥20개가 필요하다.
  미달 시 판정 불충분이며 join 목적함수 감소로 대체하지 않는다.
  WR 대비 결과도 공개해 회전 사용과 전역화의 효과를 구분한다.
- 비용: CPU만 사용. arm당 solver 최대 300 function evaluations;
  실패·미수렴은 기록하고 채택하지 않는다. GPU 추론·GS 학습은 이번 단계에서 실행하지 않는다.
- 실행·출력: `logs/r2_p1_20260909/`. 착수 시 상태: 구현 및 합성 검증 중.

#### R2-P1 결과 — 2026-09-09, 완료·미채택

구현: [`eval/window_alignment.py`](../eval/window_alignment.py),
실행·평가: [`eval/run_window_alignment.py`](../eval/run_window_alignment.py).
NumPy/SciPy/OpenCV 기반 CPU 경로이며 기존 Pi3X/3DGS 기본 실행 경로에는 연결하지 않았다.

**전역 SE(3) 하나로 ARKit에 정렬한 뒤 프레임별 위치 차이의 평균:**

| arm | `2994fa` | `7d3d52` |
| --- | --- | --- |
| W0 — 기존 depth 순차 연결 | 2.4632 m | 0.24358 m |
| WR — 회전 기반 순차 연결 | **1.5050 m** | 0.24353 m |
| WG-L — 위치＋회전 전역 최소제곱 | 1.7133 m | 0.24301 m |
| WG-R — 블록 soft-L1 전역 정렬 | 1.8901 m | 0.24304 m |

W0 대비 `2994fa`에서 WR은 38.9%, WG-L은 30.4%, WG-R은 23.3% 감소했다.
WG-R은 WR보다 25.6% 나쁘다. `7d3d52`는 사실상 동률이며 WG-R 변화는 −0.22%다.
모든 solver는 제한 내 수렴했다. WG-L/WG-R의 function evaluations는
`2994fa` 10/15회, `7d3d52` 6/7회다. 이미지 진단 포함 실행 시간은 각각 약 2.6/1.2 s였다.
ARKit 대비 차이는 절대 metric 정확도나 3DGS 개선을 입증하지 않는다.

전역화와 중복 포즈 평균의 효과를 분리하려고 **결과 후 집계 방식 진단**도 했다.
WR에 같은 평균 집계를 적용하면 `2994fa` 1.5061 m;
WG-L/WG-R에 첫 추정 유지 집계를 적용하면 1.7125/1.8905 m다.
이 사례에서 전역 arm의 열세가 중복 집계 방식만의 효과는 아니다.
이 진단은 사전등록 arm이나 채택 기준을 변경하지 않는다.

**영상 검증은 불충분하다.** 제안 pair는 49/44개였으나 기준을 통과한 것은
세션마다 1개다. mutual match 부족이 각각 35/40개, confident depth inlier 부족이
12/3개, descriptor 부재가 1/0개였다. 한 pair의 재투영이 좋아진 것을 세션의
독립 검증 성공으로 보고하지 않는다. 따라서 WG-R의 위치 조건은 통과했지만
최소 10 pair 조건을 못 채워 **확장 후보로 채택하지 않는다**.
WR도 영상 검증을 통과한 새 기본 경로로 승격하지 않는다.

추가 원인 진단에서 `2994fa` window 6 (`frame 936–1212`)은
그 window만 ARKit에 최선의 강체 정렬을 해도 평균 위치 차이 **0.481 m**가 남았다.
13개 window의 이 수치 중앙값은 0.0248 m다.
이는 **특정 window 내부 형상부터 다르다**는 근거이며, 강체 window 정렬만으로
해당 차이를 제거할 수 없다. 이 per-window 참조 정렬은 원인 진단일 뿐,
제안 방법의 성능 평가에는 사용하지 않았다.

**판단 변경:** 전역 정렬이 먼저 성공해야 R3에 갈 수 있다는 순서를 고집하지 않는다.
문제 window와 이웃의 RGB/depth 관측을 확보해 제한된 국소 BA를 시험할 준비를 한다.
별도로 재방문/외부 관측 없이 overlap의 일관성만 최적화하는 한계도 남아 있다.
이번 결과는 고정 scale·고정 window 내부 포즈의 한 pilot을 평가한 것이며,
LiDAR 측정이나 실제 재방문을 추가한 전역 최적화 일반을 반박하지 않는다.

검증은 `eval/test_window_alignment.py`, `eval/test_window_reprojection.py`,
기존 `eval/test_pi3_join.py`, fixture 생성·reader를 실행해 모두 통과했다.
합성 이상 관측 통제에서 나머지 정상 프레임의 평균 위치 오차는 일반/강건 arm이
9.38/0.21 cm였다. 이 합성 통과와 실제 세션의 연구 가설 판정은 구분한다.

재실행 명령·입력 해시·출력은
[`실험 기록`](../logs/r2_p1_20260909/README.md)에,
비교 그림은 [`trajectories.png`](../logs/r2_p1_20260909/trajectories.png)에 남겼다.

### 1.3 한 시간 연구 세션 — 2026-09-09 00:36 KST 착수

목표 시간은 약 01:36 KST까지다. 진행 기록은 `logs/research_hour_20260909/`에 둔다.
이번에는 R3 국소 RGB/depth 관측 최적화를 우선하고, 관측이 허용하면 IMU 회전과
30 Hz 자료 진단을 확대한다. 현재 데이터가 판정할 수 없는 항목은 미검증으로 남긴다.

**R3-P1 사전 조건 — 새 arm 실행 전 고정:**

- 개발 세트: `2994fa` window 6, 대조: `7d3d52` window 2.
  추가 양호 포즈 통제: `cb4586`의 normal 사진 중 48–71번째, ARKit 초기화.
- RGB 폭 960 px, LK forward/backward ≤1 px, 최소 길이 3인 트랙을 사용한다.
  track ID 전체를 seed 42의 고정 순열로 80/20 train/held-out 분리하고,
  held-out track의 RGB/depth 관측은 최적화에 넣지 않는다.
- 깊이는 native 격자에서 읽는다. ARKit confidence ≥2, 0.2–5 m,
  3×3 patch 범위 <0.1 m인 관측만 감독에 사용한다.
- B0 초기 포즈, B-RGB RGB 잔차＋약한 포즈 prior,
  B-RGBD 동일 조건＋LiDAR z 잔차를 비교한다. 두 arm 모두 포즈와 landmark를 조정한다.
  RGB-only도 동일한 depth 기반 초기값을 받으며, 순수한 무센서 모델이라고 부르지 않는다.
- 첫 카메라 포즈 고정. intrinsics는 고정. RGB 잔차 1.5 px, depth 잔차
  `0.03 + 0.01*z` m로 정규화한 블록 soft-L1. 포즈 prior는 0.5 m/30°의
  고정 모델링 선택이며, 참조 ARKit extrinsics는 초기화 통제를 제외하면 평가에만 사용한다.
- 채택 화면: held-out track ≥30, 평가가 덮는 프레임 ≥80%;
  held-out 재투영 중앙값 ≥15% 감소, held-out depth abs-rel 증가 ≤10%;
  실패 window의 ARKit 평균 위치 차이 ≥20% 감소, 대조 window 증가는 ≤10%.
  ARKit 초기화 통제에서 포즈 이동 중앙값 >2 cm이면 관측 모델의 편향을 별도 조사한다.
  모든 조건을 확인하기 전에는 기존 포즈를 교체하거나 3DGS 개선을 선언하지 않는다.
- solver 최대 150 function evaluations/arm. 같은 분할과 모델링 값을 유지하며
  실제 결과를 보고 threshold를 완화하지 않는다. 실패 원인 확인을 위한 후속 진단은 구분 기록한다.

착수 시 상태: 트랙 확보·구현 중. 완료 결과는 §1.4에 기록했다. 이전 R2-P1의 pair 부족 판정을 소급 변경하지 않는다.

00:46 KST 관측 감사: `2994fa` window 6은 RGB/depth 트랙 8개(held-out 2개),
`7d3d52` window 2는 44개(held-out 9개)로 사전 최소 표본에 미달한다.
`2994fa` frame 1080 원본은 거의 검게 기록돼 있다. 이 두 경우의 최적화는
진단으로만 남기고 채택에 쓰지 않는다. `cb4586`에서는 600개/120개를 확보했다.
원래 threshold를 완화하지 않고 같은 `cb4586` 구간의 Pi3X 초기화와,
`d06152` wide 궤적의 중앙 24장을 추가한다. 후자는 640 px 원본 폭,
confidence 미기록을 명시하고 유효 depth·경계 검사만 사용한다.
이 확장 세트는 새 BA 결과를 보기 전에 정했으며 기존 실패 판정은 유지한다.

**R4-P1 사전 조건 — IMU 회전 보조:** `cb4586` 전체 normal ARKit 포즈와
CMDeviceMotion attitude의 0.2 s 상대 회전으로 고정 camera/device 축 변환을 추정한다.
회전량 1–45°인 pair 중 5개마다 하나는 보정에서 제외한다. quaternion 정/역방향은
훈련 오차로 선택하고 선택 후 고정한다. 시간 offset은 0으로 고정하며 이번에 함께 맞추지 않는다.
보정의 제외 pair와 `2994fa`, `7d3d52`, `87bc2c`, `6db873`, `7d9994`에서
상대 회전 중앙값 ≤1°, p95 ≤3°인지 먼저 검사한다. 충분한 pair <10이면 미판정.
통과 시 같은 R3 트랙·초기값에서 인접 상대 회전 2° soft-L1 제약만 추가한다.
회전 잔차 감소 자체를 성공으로 세지 않고 held-out RGB/depth와 pose 변화로 비교한다.
CoreMotion/ARKit는 독립 raw 센서가 아니며, 이 실험은 metric scale의 관성 검증이 아니다.

**R3-U1 사전 조건 — 다른 렌즈로의 개선 전달:** `d06152` 중앙 구간의 wide-only
BA 결과를 학습에 넣지 않은 UW 사진으로 검사한다. 각 wide 시점에서 시간 차
0/0.4/1.0 s에 가장 가까운 UW 사진을 제안하되 wide pose 시간 범위 밖은 제외한다.
wide 640/UW 960 px, mutual SIFT ratio 0.75와 F-RANSAC 1.5 px를 사용한다.
wide train LK 관측에서 12 px 이내인 특징은 평가에서 제외하고 native depth로
3D 점을 만든다. UW에는 실제 시각의 body 포즈 보간＋고정 rig를 적용한다.
20개 이상 대응이 있는 pair ≥10개가 필요하다. 동일 pair에서 재투영 중앙값이
초기값보다 ≥15% 감소하는지 보며, UW 반경 >0.6 구간을 별도로 보고한다.
새로운 UW 학습이나 rig 재보정은 하지 않으므로 이것은 공동 BA 이전의 전달 검사다.

01:02 KST 중간 결과: RGBD는 wide held-out 잔차를 줄였지만 `cb4586` ARKit
초기화 통제에서 포즈 이동 중앙값 4.3 cm로 2 cm 기준을 넘었다.
`d06152` UW는 4/62 pair만 통과했고, RGBD의 중앙값은 6.39→6.58 px였다.
기존 기준을 유지한다. **R4-P2**는 자유도를 줄이는 별도 후속 진단으로,
카메라 회전을 첫 프레임에 정렬한 IMU attitude로 고정하고 위치/landmark만 조정한다.
같은 트랙·split·depth 잔차를 유지하며 threshold나 sigma sweep은 하지 않는다.
고정 회전은 uncertainty를 표현하는 최종 모델이 아니라, 잘 검증된 회전 관측의
활용 효과와 자유로운 BA의 편향을 분리하는 통제다.

#### 01:10 추가 검증 사전등록: 세션 확대와 30 Hz 관측량

- R4 보정은 `cb4586`의 기존 C와 시간 offset 0을 그대로 유지하여 모든 ARKit
  세션에 적용한다. 같은 10 pair·중앙값 1°·p95 3° 기준을 사용한다.
  보정에 사용한 세션과 나머지 세션의 결과를 분리한다.
- R3/R4 확대는 `2d9844`, `e92fe5`의 기존 wide Pi3X 궤적 중앙 24장으로 한다.
  동일 트랙 split과 RGBD/IMU 회전 고정 조건을 사용하며, 임계값은 바꾸지 않는다.
- R5-P1: `6db873`, `7d9994`, `d4fa51`, `2b38b6`의 30 Hz 사진에서 시작 3초 이후
  첫 8개의 겹치지 않는 12-frame 간격(약 0.4초)을 사용한다. 시작·끝 사진과
  시작 특징점을 같게 두고 6/3/1 frame 간격으로 LK를 이어 5/10/30 Hz를 비교한다.
  960 px, 1200 corners, 거리 12 px, LK 31 px/5 pyramid, 각 step FB 1 px,
  전체 왕복 1 px를 고정한다. native depth confidence ≥2와 기존 평탄성 기준으로
  시작점만 고른다. ARKit 상대 pose의 depth 재투영과 끝점 차이가 ≤3 px인
  관측의 비율을 시작점 전체를 분모로 계산한다. 각 session의 8 pair 중앙값과
  30 Hz−5 Hz paired 차이를 보고한다. ≥100 시작점/session과 4개 중 3개 session에서
  유효 비율 +10 percentage point가 관측량 개선의 예비 기준이다.
  광학 흐름 자체나 ARKit을 정답으로 간주하지 않으며 선명도/GS 품질 개선은 판정하지 않는다.

#### 01:14 R3/R4 적용 범위 확대 사전등록

`pi3traj` 중 6자리 session ID 이름이고 `reference`가 있는 18개 캐시 전부의
중앙 24장을 사용한다. 기존에 본 실패 구간/좋은 구간과 구별하여 별도 census로 저장한다.
각각 Pi3X 초기화의 RGBD, Pi3X 초기화의 IMU 회전 고정 RGBD,
ARKit 초기화의 IMU 회전 고정 RGBD 대조군을 같은 관측으로 비교한다.
보정 C, split, residual, 150 evaluation 상한과 30 held-out tracks/80% coverage,
재투영 15%/depth 악화 ≤10%, ARKit 대비 위치 차이 20% 감소,
ARKit 대조군 원래 gauge에서 이동 중앙값 ≤2 cm 기준을 유지한다.
수렴·표본·대조군 조건을 모두 통과한 세션 수를 전체/평가 가능한 세션 수와 함께 보고한다.
ARKit은 독립 metric GT가 아니며, 이 검사는 기본 경로 채택의 충분조건이 아니다.

#### 01:18 중간 판정과 후속 진단

18개 중앙 구간 중 기존 관측량 기준을 충족한 것은 9개이며, 회전 고정 arm의
모든 예비 기준을 통과한 것은 3개다. RGB 재투영은 18개 모두 15% 이상 좋아졌지만
ARKit 대비 위치 차이 20% 개선은 4개뿐이다. residual 감소만으로 metric 개선을
판정할 수 없다는 근거로 남긴다. 다음 진단은 기존 판정을 변경하지 않는다.

- training track의 camera 연결 성분을 조사한다. 첫 카메라에 고정한 gauge가
  어느 관측까지 전달되는지, 완전히 관측되지 않은 camera가 있는지 보고한다.
- 같은 held-out track의 인접 관측에서 RGB-only Sampson 거리를 계산한다.
  초기 baseline ≥1 cm인 pair를 먼저 고정하고 모든 arm 및 ARKit을 같은 pair로
  평가한다. LiDAR anchor를 제거했을 때의 영상 일관성 진단이며 절대 scale은 검사하지 못한다.
- 30 Hz 네 세션의 같은 평가 구간에 대해 기존 `motion_blur.streaks`의 centred
  exposure 모델로 예측 blur를 계산한다. native 해상도와 960 px 환산을 모두 적는다.
  이는 이미지가 선명하다는 검증이 아니라 노출 중 운동량 추정이다.

#### 01:18 연결성 진단 결과와 R3-P2 prior 대조 사전등록

training 관측이 24 camera 전체를 하나로 연결한 구간은 18개 중 7개뿐이었다.
기존 예비 통과 3개 중 `f0d073`도 4개 성분으로 나뉜다. 기존 3/18 판정은 보존하고,
향후 채택에는 관측 연결성을 추가한다. 기존 기준과 완전 연결을 함께 충족한 것은
`40ebb3`, `5b88f7` 2개다. IMU 회전 고정이 translation의 미관측을 해결하지는 않는다.

R3-P2는 같은 18개에 회전 고정 RGBD＋인접 Pi3X translation increment prior를
한 조건만 추가한다. prior는 world 좌표의 인접 translation 차이, soft-L1 5 cm이며
센서에서 추정한 불확실성이 아닌 규제 강도다. 기존 0.5 m pose prior는 유지한다.
ARKit 초기화 대조군은 ARKit increment를 같은 강도로 보존한다.
기존 관측 연결성 gate는 그대로 유지한다. prior로 연결한 것을 실제 관측으로 세지 않는다.
같은 수렴/관측/영상/depth/위치/대조군 기준과 위치 차이 증감의 세션 수를 비교하며,
다른 강도는 탐색하지 않는다. 목적은 미관측 구간의 자유로운 이동을 줄일 수 있는지의 진단이다.

#### 01:20 통과 구간의 시간 블록 반복 사전등록

관측 완전 연결과 기존 기준을 모두 만족한 `40ebb3`, `5b88f7`에서 중앙 구간과
겹치지 않는 첫 24장·마지막 24장을 추가한다. 같은 회전 고정 RGBD와 ARKit
초기화 대조군을 동일 기준으로 실행한다. 별도 capture 반복이 아니라 같은
capture 내부의 시간 블록 검증이다. translation prior arm은 중앙 census의
통과 수를 늘리지 못했으므로 반복 실험에는 추가하지 않는다.

#### 01:23 R3-S1 깊이 샘플링 민감도 사전등록

native depth를 floor pixel에서 읽으면 경사진 평면에서 subpixel ray와 depth가
어긋날 수 있다. 합성 평면에서 inverse-depth bilinear 보간을 검사한 후,
18개 중앙 구간의 기존 트랙·split·유효 depth mask를 고정하고 depth 값만 교체한다.
회전 고정 RGBD와 ARKit 대조군을 다시 계산하며 기존 floor arm을 보존한다.
추가 픽셀/트랙을 선별하지 않는다. 이것은 측정 모델 민감도 검사이고 센서 scale 보정이나
정확도 증명은 아니다. 세션 통과 수와 위치 차이, 대조군 이동, 같은 RGB 트랙의
재투영 변화로 비교한다. 보간 변경으로 이전 실패가 설명되는지도 함께 적는다.

### 1.4 한 시간 실험 결과와 현재 판단 — 2026-09-09

실제 작업 구간: **00:36:37–01:35:09 KST, 약 59분**.

상세 명령, 각 arm, 입력/코드 해시, 표와 그림은
[로컬 연구 기록](../logs/research_hour_20260909/README.md)과
[집계 JSON](../logs/research_hour_20260909/summary.json)에 있다.
원본을 변경하지 않고 CPU로 수행했다. 157 optimizer arm은 같은 자료의
대조·민감도 조건을 포함한 실행 수이며 독립 장면 수가 아니다.

| 질문 | 확인한 범위와 결과 | 판정 |
| --- | --- | --- |
| camera와 IMU 상대 회전축을 공통으로 쓸 수 있나 | `cb4586`에서 보정한 C, offset 0을 54 ARKit 세션에 적용. 보정 외 53개 포함 전부 ≥10 pair/중앙값 ≤1°/p95 ≤3° 통과. 세션별 p95 최댓값 <0.694° | 회전 제약 구현의 근거 확보. 처리된 센서 추정치 간 일관성이며 독립 GT 아님 |
| RGBD BA가 Pi3X 위치 형상을 개선하나 | 18개 캐시 중앙 24장. RGBD의 ARKit 대비 위치 차이는 15/18개에서 증가. IMU 회전 고정 후에도 10/18개 증가. 고정 arm의 RGB 재투영은 18/18개에서 ≥15% 감소 | residual 감소만으로 metric 개선 판정 불가 |
| 평가에 충분한 관측이 있나 | track ≥30/coverage ≥80%는 9/18개. 실제 training camera가 전부 연결된 것은 7/18개 | coverage 외에 관측 연결성 추가 필요. 완전 연결도 충분한 시차를 보장하지 않음 |
| 좋은 구간은 반복되나 | 원래 예비 기준 통과 3/18, 후속 연결성까지 만족한 중앙 구간은 `40ebb3`, `5b88f7` 2개. 두 세션의 첫/마지막 24장 4개는 전체 기준 0/4 통과 | 특정 구간 성공을 세션 전체로 일반화하지 않음 |
| Pi3X translation prior나 depth 보간으로 실패가 해결되나 | 인접 이동량 5 cm prior, inverse-depth bilinear를 각각 고정 조건으로 비교. 관측 연결성과 기존 기준을 함께 통과한 구간은 동일 2개 | 단일 규제 강도/샘플링 방식으로 전체 실패가 설명되지 않음 |
| wide 개선이 UW로 전달되나 | `2d9844`: wide 78.3%, UW 13.3% 감소(56 pair). `e92fe5`: wide 43.2%, UW 8.6%(47 pair). `d06152`: wide 35.9% 감소, UW 2.9% 악화(4 pair, 부족) | 전체 UW 15% 개선 기준 미충족. 두 렌즈의 nuisance 분리 필요 |
| UW cone 바깥도 개선될 가능성이 있나 | radius >0.6 관측: `2d9844` 51개, 14.33→7.59 px; `e92fe5` 45개, 9.27→6.64 px | 일부 지지. 독립 landmark/scene 수가 아니고 전 FOV 판정에는 부족 |
| 30 Hz면 연결성과 선명도가 충분해지나 | 4세션×8개 같은 0.4초 endpoint. 유효 대응 비율의 30−5 Hz paired 중앙값 +7.47/0.00/4.95/1.06 pp. 예측 노출 운동 960 px에서 6.54/3.10/4.78/2.79 px | +10 pp, 3/4세션 기준 미충족. 일부 pair 이득은 있으나 고빈도=sharp GT 아님 |

**기존에 지정한 실패 구간의 판정도 보존한다.** `2994fa` window 6은 사진이 매우 어두워
8 track/2 평가 track뿐이다. IMU까지 추가해 재투영이 크게 줄어도 ARKit 대비 위치 차이
약 48 cm는 남았다. `7d3d52` window 2도 평가 track 9개로 부족했고 위치 차이가 악화됐다.
`cb4586`의 처음 지정한 풍부한 구간은 회전 고정으로 1.99→0.80 cm 차이 감소를 보였지만,
ARKit 대조군 이동 2.015 cm가 2 cm 기준을 넘었다. 뒤의 중앙 census는 다른 구간이고
같은 개선을 보이지 않았다. 좋은 구간의 결과만 대표값으로 쓰지 않는다.

**평가의 범위를 제한한다.** 여기서 holdout은 BA optimizer가 보지 않은 track이다.
Pi3X 캐시는 같은 사진으로 만들어졌고 conditioning/revision의 완전한 이력도 없으므로
§6의 엄격한 프레임 단위 전체 파이프라인 holdout을 충족한 것은 아니다.
ARKit 비교는 한 번의 SE(3)만 맞추며 scale은 재추정하지 않는다. camera/device 축 변환의 세션 간 검증은 ARKit 모드에서 수행했으며 MultiCam으로의 적용은 추가 검증이 필요하다.
ARKit/CMDeviceMotion의 공유 센서와 LiDAR 측정 오차 때문에 이것을 독립 절대 정확도로 해석하지 않는다.
이번 연구에서 새 GS·외부 치수 검증을 하지 않았으므로 metric 보장/photorealism 개선을
달성했다고 판정하지 않는다.

**방향 수정:** 다음 R3는 센서를 한 번에 더 강하게 최적화하는 방식에서,
관측으로 연결된 구간의 제한된 자유도와 조건별 측정 오차를 먼저 검증하는 방식으로 진행한다.
IMU는 회전·노출 중 운동에, Pi3X는 초기 형상과 미관측 부분의 명시적인 prior에 사용한다.
다른 렌즈와 외부 치수를 통해 그 prior가 실제로 맞는지는 별도로 검증한다.

검증: 새로운 BA/회전축/시간 간격/epipolar·연결성 self-test와 기존 window/투영/join/export
self-test 8개, fixture reader 1개를 통과했다. 합성 평면의 inverse-depth 보간도 검증했다.

### 1.5 R3-O1 관측 정보량·분할 안정성 — 2026-09-09 06:33 KST 착수

이전 중앙 18개와 반복 4개 구간을 모두 사용한다. 새 실험은
[사전등록](../logs/observability_20260909/protocol.json)에 고정했으며 기존 결과는 보존한다.

- 기존 Pi3X 초기화·고정 IMU 회전·같은 residual/split을 사용한다.
- RGB/depth measurement Jacobian에서 landmark를 Schur 소거하여 translation의
  정보 행렬을 만든다. pose prior·IMU·cheirality penalty는 실제 측정 정보로 세지 않는다.
  회전을 조건으로 한 수치적 민감도이며 완전한 posterior covariance가 아니다.
- training track ID를 정렬해 홀/짝 두 묶음으로 나누고 각 묶음만으로 BA를 실행한다.
  원래 held-out track은 양쪽에서 모두 제외한다. 전체·양쪽 모두 camera 관측이
  연결돼야 하며, 조건부 camera 3D std 최댓값 ≤2 cm, 두 결과의 동일 gauge 내
  translation 차이 중앙값 ≤1 cm/p95 ≤3 cm를 사전 예비 기준으로 삼는다.
- ARKit 위치나 원래 평가 track은 이 정보 계산·분할 선택·최적화에 사용하지 않는다.
  이 기준과 이전 held-out/ARKit 진단의 관계를 보고하며 임계값을 사후 조정하지 않는다.
- 합성 기하에서 Jacobian, Schur 소거, gauge/nullspace, depth 유무, held-out 격리,
  공통 depth scale bias의 비검출 한계를 먼저 확인한다.

#### R1-F1/R3-F1 사진 단위 제외 실험 사전등록

관측 연결성이 있었던 중앙 7개에서 IMU 보정에 쓴 `cb4586`을 제외한 6개
(`40ebb3`, `5b88f7`, `5bd1ed`, `683ef1`, `7d3d52`, `841e94`)를 사용한다.
24장 중 0-based index 3/9/15/21의 RGB/depth를 Pi3X conditioning과 BA에서 모두 뺀다.
20장만으로 pinned Pi3X를 다시 추론하고 dense predicted depth/confidence도 캐시한다.
기존 캐시와 수치를 직접 비교하지 않고 새 Pi3X와 그 초기화의 고정 IMU RGBD를 비교한다.

- 입력은 training RGB/K/confidence ≥2 native LiDAR, pose conditioning 없음.
  pixel limit 255000, seed 42, metric scale은 training depth로만 추정한다.
- 평가 camera는 첫 training camera의 Pi3X↔ARKit rigid gauge 하나로 고정한다.
  BA 전후에 동일한 ARKit 평가 camera를 사용하고 arm별 pose/scale refit은 하지 않는다.
- 각 제외 사진에 대해 직전/직후 training 사진 각 2장을 제안한다.
  mutual SIFT 0.75/F-RANSAC 1.5 px, source depth/평탄성 검사를 통과한 ≥20점/pair,
  source가 BA training track과 ≥12 px 떨어진 대응을 사용한다. 양쪽 영상은 960 px.
  ≥10 pair·평가 camera ≥3개, RGB 오차 ≥15% 감소·depth abs-rel 악화 ≤10%를 예비 기준으로 한다.
- 구간 선택에는 이전 전체 영상 결과를 사용했으므로 개발 세트의 frame holdout이며,
  보지 않은 세션의 확증 실험이라고 부르지 않는다. ARKit 평가 pose도 독립 metric GT는 아니다.
- [정확한 frame ID·설정](../logs/observability_20260909/frame_holdout/protocol.json)을 저장했다.

#### R3-G1 제한된 GS 포즈 대조 사전등록

R3-F1에서 ≥10 pair와 4개 평가 camera를 확보한 두 세션 `5b88f7`, `841e94` 모두에
Pi3X/BA 포즈 대조 학습을 진행한다. 양호한 개선 사례를 고른 것이 아니며 BA 채택과도 별개다.
20 train/4 eval, 960×720, seed 42, 4000 iteration, downscale schedule 1000으로
2000 step부터 원 해상도를 사용한다. DN-Splatter의 depth L1 λ0.2, depth-normal λ0.05,
camera optimization off를 고정한다. 같은 세션의 양쪽 arm은 Pi3X training LiDAR로 만든
동일 초기 점군을 사용하여 학습 camera pose만 바꾼다. 평가 camera는 R3-F1과 같고
평가 RGB/depth는 Pi3X·BA·초기 점군·GS gradient에 사용하지 않는다.

4000-step checkpoint만 평가하며 held-out score로 checkpoint/설정을 선택하지 않는다.
세션별 paired PSNR ≥3/4 view 개선 및 평균 +0.5 dB 이상을 외관의 예비 지지 기준으로
사용하고 SSIM/LPIPS와 depth 일관성도 함께 보고한다. 짧은 예산의 포즈 대조이므로
기존 7000-step scene 성능과 직접 비교하거나 photorealism/절대 metric 개선을 선언하지 않는다.
[정확한 설정](../logs/observability_20260909/gs_pilot/protocol.json)을 보존한다.

R3-G1 실행 조건 수정(완료된 학습 step/평가 결과를 보기 전): DN-Splatter의 depth-normal
경로가 multi-resolution일 때 691200/43200 pixel 격자 불일치로 첫 실행에서 중단됐다.
외부 학습기 코드는 수정하지 않고 **모든 arm을 num_downscales=0, 960×720 전체 해상도**로
통일한다. seed/4000 step/loss/평가 기준은 그대로이며 실패 로그와
[수정 명세](../logs/observability_20260909/gs_pilot/protocol_fullres.json)를 함께 보존한다.

R3-F2 평가 범위 보완: R3-F1은 이전 track holdout에서 쓰던 source 특징점의 training
track 12 px 배제를 그대로 적용해 표본이 적었다. 이번에는 target 사진이 upstream에서
완전히 제외돼 있으므로 **training landmark를 새 사진에 투영하는 평가도 유효**하다.
제안 pair·매칭·깊이·기준을 유지하고 source 거리 배제만 0 px로 바꾼 결과를 별도로
보고한다. F1 판정과 이미 정한 GS 세션 선택은 바꾸지 않는다.
[변경 변수와 이유](../logs/observability_20260909/frame_holdout/protocol_source_all.json)를 남긴다.

#### 완료 결과와 다음 판별 실험

[전체 실행 기록·표·그림](../logs/observability_20260909/README.md)에 개별 관측,
실패 로그, 추론/학습 명령과 checkpoint를 남겼다.

- **O1: 4/22구간만 training 관측 기준 통과.** `cb4586`은 조건부 std 최댓값
  1.37 cm인데 training track 절반을 바꾸면 위치가 중앙값 5.43 cm 달라졌다.
  국소 정보 행렬은 최적화의 실제 민감도를 충분히 설명하지 못한다. 합성에서는
  공통 depth scale bias 5%도 완전한 rank/작은 std/정확한 fit을 유지했다.
  따라서 이 값을 절대 scale 보증이나 보정된 신뢰구간으로 사용하지 않는다.
- **F1/F2: 새 Pi3X 추론 6개 완료, 공동 기준 통과 각각 0/6.** 평가 사진/depth를
  입력부터 제외했으며 평가 camera를 arm별로 다시 맞추지 않았다. F2 `40ebb3`의
  재투영은 19.92→5.35 px로 개선됐지만 depth abs-rel은 0.0088→0.0102로 악화했다.
  새 20-frame training screen은 `5bd1ed`만 통과했으나 평가 표본이 부족했다.
- **G1: 4000-step GS 4개 학습·고정 평가 완료.** 같은 초기 점군과 평가 camera를
  쓰고 training pose만 바꿨다. 두 세션 모두 PSNR 개선 시점은 2/4로 예비 기준 미달이다.

| 세션 | Pi3X → BA PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | LiDAR RMSE cm ↓ | depth abs-rel ↓ |
| --- | --- | --- | --- | --- | --- |
| 5b88f7 | 16.54 → 16.29 | 0.611 → 0.599 | 0.332 → 0.334 | 5.98 → 4.72 | 0.0402 → 0.0387 |
| 841e94 | 17.45 → 17.27 | 0.616 → 0.615 | 0.202 → 0.190 | 2.24 → 2.06 | 0.0120 → 0.0146 |

표는 평가 4장의 metric 평균이다. depth는 confidence ≥2의 LiDAR를 같은 metre 단위로
비교하며 scale 재정렬을 하지 않았다. 저장한 float RGB/depth에서 수치를 재계산했고,
동일 평가 사진·camera와 상류 aggregate metric 일치를 검증했다.
전체 [5b88f7 렌더](../logs/observability_20260909/gs_pilot/5b88f7_all_views.png)와
[841e94 렌더](../logs/observability_20260909/gs_pilot/841e94_all_views.png)에는 모서리,
책/화면 글자와 반사 영역의 번짐·변형이 남는다. 원본 사진에도 흐림이 있으며,
단일 seed·짧은 구간·4000 step 결과로 photorealism의 달성/불가능을 판정하지 않는다.
ARKit 평가 pose와 LiDAR도 독립 ground truth는 아니다.
또한 공통 초기 점군은 Pi3X pose에서 만들었으므로 초기 정합은 Pi3X arm에 유리할 수 있다.
이 대조가 판별하는 것은 **동일 점군에서 training pose만 교체한 효과**다. BA pose로
점군까지 다시 만드는 전체 파이프라인의 효과나 충분히 긴 학습의 수렴 성능은 판별하지 않았다.

**연구 판단:** scale이 주어진 초기화에 RGBD BA와 고정 IMU를 적용하는 현재 조합은
기하·외관의 공동 개선을 입증하지 못했다. 그렇다고 융합 자체를 기각한 것은 아니다.
현재 대조는 회전 교체와 translation 최적화를 동시에 바꾸므로 악화 원인을 분리하지 못한다.
관측 정보량은 실패를 거르는 진단으로 유지하고, 다음 작업 순서를 다음처럼 구체화한다.

1. **기존 줄자 세션의 독립 길이 검사:** 알려진 1 m/2 m 구간의 양 끝점과
   가시성·불확실성을 고정한 뒤 native LiDAR와 새 Pi3X의 길이를 별도로 평가한다.
   길이를 scale fitting에 쓰지 않고, 거리·표면각에 따른 편향을 보고한다.
   한 선분 통과를 장면 전체 metric 보증으로 확대하지 않는다.
2. **제약별 ablation:** 다른 시간 블록에서 평가 사진을 먼저 제외하고 Pi3X,
   회전만 IMU로 교체, RGBD BA, soft relative-IMU BA를 같은 관측으로 비교한다.
   깊이의 공통 bias와 회전 고정 오차를 residual 분포·track 분할 안정성으로 구별한다.
   이번 여섯 중앙 블록은 개발 세트로 남기며 재사용 결과는 탐색이라고 표시한다.
   기하의 지지가 확보되면 pose/초기 점군을 교차한 2×2 GS 대조로 초기 정합 효과도 분리한다.
3. **외관 오차 분리:** 기하를 고정한 조건에서 노출/blur와 반사·비정적 영역의 오차를
   따로 측정한다. 예산 증가나 appearance 모델 변경의 효과를 포즈 변경 효과와 섞지 않는다.
   관측 범위 밖 UW 확장과 장면 전체 재구성은 이 판별 결과를 받은 뒤 확대한다.

### 1.6 우선순위 작업 목록 — 2026-09-16, LiteReality-Agent 분석 반영

#### LiteReality-Agent 에서 확인한 것

`~/uv_workspace/LiteReality-Agent` (44k 줄, 2026-08-01 시작, 09-12 마지막 커밋).
파이프라인: 스캐너 앱 캡처(`frame_N.json` 의 `cameraPoseARFrame`·`intrinsics`, `depth_N.png`
uint16 mm 256×192, `conf_N.png`, `pointcloud.pcd`, RoomPlan `room.usdz`) → RoomPlan usdz 파싱
(벽·바닥·문·창 + 오브젝트 OBB/카테고리) → 박스 병합(벽 세그먼트를 가로지르는 엣지 거부) →
레이아웃 수리(개선이 증명될 때만 적용, 삭제 금지) → 오브젝트별 뷰 선택(depth occlusion 0.1 m,
size×centredness×visibility) → GroundingDINO crop 정제 → 참조 이미지 생성 → TRELLIS 또는 절차적
생성 → 물리 속성(occupancy 질량, 재료 마찰, CoACD collider, 관절 spec) → Room.py → 에이전트
저작(render|photo 같은 ARKit 포즈, critic, Poly Haven 재질) → collision/support/validation 게이트
→ MuJoCo/URDF/MJCF, `--shake` 로 5 s 드리프트 측정(7~1234 mm).

시뮬레이터가 스캔에서 필요로 하는 것을 기준으로 두 저장소를 놓으면:

| 필요한 것 | LiteReality-Agent | 우리 |
| --- | --- | --- |
| 방 껍질(벽·바닥·개구부, metric) | RoomPlan → `SHELL`, 개구부를 빼서 jamb/lintel 박스 | `planes.jsonl` 뿐 |
| 오브젝트를 개별 body 로 분리, 충돌 형상, 질량·관성·마찰, 관절 | 있음 — 단 질량은 추정, 관절은 카테고리 spec | 없음 |
| 오브젝트 형상·외관의 실물 정확도 | 생성물(TRELLIS crop 1장, OBB 에 비균일 스케일). 픽셀 지표 없음 | LiDAR·다중뷰 기하, 3DGS PSNR |
| metric scale 독립 검증 | 없음 — ARKit 신뢰. shake 1234 mm 의 원인(의자가 창틀 안 44 mm)을 상류로 넘김 | 줄자·보드·join 스케일 분해 |
| 포즈 | ARKit 그대로, ~1 fps 키프레임 | ARKit + Pi3X + BA 연구, 5~30 Hz, 두 렌즈 |
| 시뮬레이터 export·검증 게이트 | 있음 | 없음 |

저쪽은 물리·의미 층을, 우리는 관측 충실도 층을 갖고 있고 입력·좌표 규약이 같다. 결합 형태는
real2sim 표준 — 물리는 메시 body, 시각은 3DGS, 한 metric 좌표계 — 이고, 저쪽 export 가
시각 메시를 collision 에 안 쓰는 visual-only geom 으로 이미 분리해 두어 그 자리에 우리 splat 이
들어간다. 제약: RoomPlan 은 ARKit 세션 위에서만 돌므로 MultiCam/UW 경로와 공존 불가 —
**UW 는 외관 층 전용**이다. `RoomCaptureSession(arSession:)` 으로 우리 `ARRecorder` 와
공존하는지는 미확인이며 그것이 S2 다.

가져올 것: RoomPlan 껍질(R7 의 가장 싼 경로), sim-ready 계약(`doc/Sim-Ready-intergration/Mujoco.md`
의 `SHELL`/`room_layout`/visual/`physics.json`, body 4종 structure/articulated/free/attached),
QC 게이트 원칙(실 삼각형로 접촉 판정·convex 분해로 거리만, 못 고치는 것은 보고만), 벽 단위
커버리지(`select_views` set-cover, `surface_views` blur·coverage %, `stitch_wall` unknown mask),
바닥 z 추정(최저 밀집 15 cm 구간의 최강 bin), 브라우저 뷰어 골격(`room_ops/walk`).
가져오지 않을 것: 생성 자산, 재질 fetch, 에이전트 하네스 — 그리고 포즈·스케일에 관한 것은 없다.

#### 우선순위 표

전제: "두 층이 우리 캡처 위에서 붙는가"(S1~S3)가 한 주 안에 답할 수 있는 가장 큰 미지수라
1순위다. R7 은 BA 없이도 가는 층이므로 R3 연구보다 앞에 둔다. 선행 조건이 없고 검증 자료가
이미 있는 것(G, V1, V2)은 그 사이에 끼운다. 상태는 §1 표에서 갱신하고 이 표는 순서만 적는다.

| 순위 | ID | 작업 | 근거 | 완료 판정 | 선행 · 비용 |
| --- | --- | --- | --- | --- | --- |
| 0 | P0 | 미커밋 작업 커밋(23 eval 스크립트, 3DGS.md, 이 문서) | 3 주치가 커밋 없이 쌓임 | 커밋 | 없음 · 반나절 |
| 1 | S1 | 우리 세션 → LiteReality 캡처 포맷 변환기 `tools/export_litereality.py` | 규약·해상도 동일. 저쪽 파이프라인·QC·뷰어를 우리 데이터에 그대로 돌림 | 저쪽 `evidence_kit.read_scan.Scan` 으로 읽히고 `scripts/capture/build_pointclouds.py` 가 클라우드를 냄. 포즈·K 왕복 오차 0 | 없음 · 하루 |
| 2 | S2 | RoomPlan 공존 측정 — `RoomCaptureSession(arSession:)` 으로 `ARRecorder` 세션 공유 | RoomPlan 이 벽·오브젝트 body 를 촬영 중에 줌. 저쪽 파이프라인은 usdz 필수. 공존 미확인이 전체 게이트 | 같은 걸음 on/off 2 arm: 비디오 포맷 유지, 프레임 드롭 0, ARKit 포즈 차·depth 신뢰도·열을 실행 전에 고정한 기준으로 | 앱 빌드(CI) · 2 일. 촬영 동작만 바꾸고 계측기는 안 건드림 |
| 3 | S3 | LiteReality 파이프라인을 우리 캡처 1개에 끝까지 (`--through simulate --from-seed`) | "붙는가"의 직접 답. `--shake` 가 정량 지표 제공 | MuJoCo 로드, 5 s 드리프트 mm, 검출/실제 오브젝트 수, `export_report.json` fallback 수 | S1+S2 · 하루 (Modal 또는 로컬 GPU) |
| 4 | G | 줄자 세션(1 m/2 m ×3) 독립 길이 검사 — native LiDAR vs Pi3X, 스케일 피팅에 안 씀 | §1.5 다음 작업 1. 데이터·CPU 만으로 가능. 저쪽엔 없는 자리 | 길이 편향 % 와 거리·표면각 의존 보고. 한 선분을 장면 전체로 확대 안 함 | 없음 · 반나절 |
| 5 | S4 | R7 산출물을 sim-ready 계약으로 정의 | 저쪽 `Mujoco.md` 가 검증된 계약. "mesh 생성"이 아니라 "시뮬레이터가 받는다"가 판정 | §2 표 제3축 = MuJoCo 로드 + shake 드리프트 상한 + collision/support 게이트 | S3 · 반나절 |
| 6 | V1 | 온디바이스 블러 미터 + 휘도 | 6~12 px 가 포즈·depth·노출만으로 예측됨(3DGS.md). `0d6306` 은 휘도 0.046 에서 죽었고 화면에 신호 없음 | 먼저 `eval/build_vis.py` 식 산점도로 54 세션 inventory 대비 추적 확인, 그 뒤 한 줄 HUD | 없음 · 오프라인 반나절 + 앱 하루 |
| 7 | S5a | 3DGS ↔ RoomPlan 껍질 정렬 검증 | 하이브리드 씬의 전제. 같은 ARKit 프레임이라 fit 이 아니라 check | 벽 평면 대 splat 잔차, 두께 계측기 분해능 0.7 cm 기준 | S2+S3 · 하루 |
| 8 | V2a | 커버리지 BEV 1 단계: 표면까지 range 색칠 + 목표 1.5 m | range 가 3DGS 커버리지의 가장 센 지렛대(−0.63). 회전 훑기는 0 — 지시 금지 | 42 세션 커버리지 표 대비 라이브 추정 일치 | 없음 · 이틀 |
| 9 | V2b | 정지 직후 요약(루프 거리·커버리지·블러·휘도·depth 신뢰도) | 제일 비싼 실패 = 몇 시간 뒤 발견(0.535 m 루프, 어두운 세션) | `session-triage` 규칙이 폰에서 같은 판정 | V1·V2a · 하루 |
| 10 | J | 블러 3-arm 실험 (3DGS.md 에 사전등록, 미실행) | 게이트·판정(16/21, 25/33, 21/27) 이미 있음 | 사전등록 그대로. 음성 팔이 움직이면 실패 | GPU 하룻밤 · 실행만 |
| 11 | H | 제약별 ablation: 새 시간 블록, 평가 사진 먼저 제외 → Pi3X / 회전만 IMU / RGBD BA / soft-IMU BA | §1.5 다음 작업 2. 지금 대조는 회전 교체와 translation 을 같이 바꿈 | 기존 기준(재투영 15 %, depth ≤10 %, ARKit 위치 20 %, 대조군 ≤2 cm) 유지 | CPU · 이틀 |
| 12 | S5b | 하이브리드 씬 조립: 저쪽 collision body + 우리 3DGS visual | real2sim 표준 구성 | 시뮬레이터에서 우리 splat 렌더 + 저쪽 body 충돌이 한 좌표계 | S3+S5a · gsplat 뷰어/렌더러 · 3 일 |
| 13 | V2c | RoomPlan 오브젝트·벽 BEV + 오브젝트별 "몇 면에서 봤나" | 저쪽 `object_view_quality` 를 라이브로. 시뮬레이터가 body 단위로 필요 | 저쪽 파이프라인 누락 오브젝트 수 감소 | S2 · 이틀 |
| 14 | I | 외관 오차 분리(기하 고정: 노출/블러/반사) | §1.5 다음 작업 3. visual policy 의 sim2real gap | 포즈 변경 효과와 섞지 않은 PSNR 분해 | H · 이틀 |
| 15 | V3 | 브라우저 뷰어 골격(저쪽 `walk/` 서버 + camera strip/compare, three.js 로컬) | PNG 뿐. 궤적 ARKit/Pi3X/BA·프러스텀·클라우드·render\|real 을 한 화면에 | `trajectories.png` 류 대체 | 없음 · 이틀 |
| 16 | V2d | ARKit `sceneReconstruction` 메시 오버레이 + 본 방향 수 색칠 | 가장 싼 "무엇이 찍혔나"이자 R7 메시 후보 | ARKit 부하·열 측정 후 | S2 · 이틀 |
| 17 | Q | ARKit 경로 1920×1440 으로 보드 한 take(롤링셔터) | 6 ms 미만으로 바운드만 됨. 이 포맷의 타깃 촬영이 코퍼스에 없음 | readout ms 확정 | 촬영 1 회 |
| 18 | R | `d06152` COLMAP 제3 arm(UW 재방문 1144 inlier 확인) | UW 에서 Pi3X 잔차의 99.99 % 가 포즈. `pycolmap` 미설치 | free/depth 대비 두께·join | GPU · 하루 |

내려간 것: MultiCam/UW 전반(0.7 dB wide 손실, UW 카메라 모델, 두 렌즈 BA 전달)은 외관 층 전용이라
S5b 이후에 다시 본다. R2 는 미채택 유지, 재개 조건 없음. 세션 병합·흐린 프레임 제거·재방문
드리프트는 닫힌 채로 둔다.

규율: HUD 신호는 외생(포즈·depth·노출·기하)만 쓰고 폰에 올리기 전에 알려진 결과의 세션에서
추적 여부를 본다(추정기 잔차는 r = −0.057 이었다). 촬영 동작과 계측기를 한 커밋에 안 바꾼다.
화면은 가로 고정·스크롤 없음·신호당 한 줄·최악 하나만.

## 2. 목표와 성공의 의미

**목표:** 멀티 카메라 영상과 LiDAR·IMU·visual SLAM·Pi3X를 융합하여 관측된 정적 실내
공간의 실제 단위와 공간적 일관성을 유지하고, 촬영 경로 밖의 검증된 시점에서도
PSNR과 시각적 품질이 높은 3DGS로 재구성한다. 같은 metric 좌표계에 시뮬레이터가
로드하는 sim-ready 기하(방 껍질, 개별 body, 충돌 형상)를 함께 만든다.
관측하지 않은 표면의 복원까지 보장하는 목표는 아니다.

네 축을 따로 판정하며 하나의 종합 점수로 합치지 않는다.

| 축 | 측정할 것 | 이것만으로는 대체할 수 없는 지표 |
| --- | --- | --- |
| 절대 metric 정확도 | 학습에 쓰지 않은 길이·표면 기준의 오차와 공간별 편향 | 단위가 metres인 출력, LiDAR 학습 잔차 |
| 전역·국소 정합 | 독립 트랙 재투영, 연결부·재방문·주변부 일관성 | join 잔차 하나, 루프 끝점, gravity 잔차 |
| 외관 | 선명한 고정 평가 뷰와 연속 이동 렌더의 질감·윤곽·floaters·깜빡임 | 흐린 타깃의 PSNR, 평균 영상 대비 개선량 |
| sim-ready 기하 (2026-09-16 까지 "collision mesh — 보조 목표") | 시뮬레이터 로드, 정지 안정성(shake 드리프트), collision/support 게이트, 동일 좌표계에서 3DGS 와의 표면 정합, 누락·거짓 장애물 | 렌더 PSNR, mesh 파일 생성 성공만으로 판정. 수치 상한은 S4 에서 S3 결과를 보기 전에 고정 |

`metric 보장`은 보정·관측 조건과 검증된 오차 범위를 명시한다는 뜻으로 사용한다.
현재 결과로 모든 장면의 1% 정확도를 주장하지 않는다.
R1에서 기준 측정의 오차와 용도에 필요한 허용치를 확인하고, **새 arm의 결과를
보기 전에** 수치 기준을 고정한다. 기준이 정해지기 전에는 최종 목표 달성을 선언하지 않는다.

## 3. 출발점: 성과와 해석의 범위

아래 수치는 기존 기록의 요약이며 이번 계획 작성 중 새 학습으로 재현한 값은 아니다.

| 관찰 | 유지할 결론 | 확대하지 않을 결론 |
| --- | --- | --- |
| `d06152`: free→depth 연결에서 렌더 깊이 중앙 오차 6.53→4.72 cm, PSNR 16.51→16.76 | metric scale을 자유로운 seam scale로 다시 버리지 않는 것이 유효했다 | 구간 내부 변형·장거리 scale drift까지 해결됐다 |
| `cb4586`: 깊이·법선 감독 on/off에서 depth abs-rel 0.0108/0.1358 | LiDAR 감독은 깊이 일관성에 크게 기여했다 | 절대 형상 정확도나 미세 질감이 같은 비율로 개선됐다 |
| ICP＋Pi3X 보간은 Pi3X 단순 보간보다 거의 이득이 없었다 | 해당 결과 수준의 rate fusion은 우선순위가 낮다 | 측정 수준의 LiDAR·관성 융합도 무효다 |
| `d06152`: 두 렌즈 학습 시 같은 wide 홀드아웃에서 약 0.7 dB 손해 | 현 공동 학습 경로는 추가 관측을 품질로 전환하지 못했다 | 초광각의 정보가 본질적으로 쓸모없다 |
| 세션 병합은 같은 reference 홀드아웃에서 2.25 dB 손해, 깊이는 거의 동률 | 기존 영역에서 병합의 이득이 입증되지 않았다 | 추가 영역도 무익하다거나, 포즈 오차가 원인에서 제외됐다 |

근거: [스케일과 렌즈 실험](3DGS.md#pi3x-metric-scale-keep-it-at-the-seams),
[깊이 감독](3DGS.md#does-depth-supervision-pay--undecided-on-pixels-decisive-on-geometry),
[rate fusion](POSE.md#the-rate-fusion-buys-nothing-measured-against-the-arm-it-was-missing),
[병합](3DGS.md#merging-trains-worse-and-the-reason-is-not-the-merge).

해석을 수정해야 할 지점:

- `uw_selfcal`의 자유로운 SE(3) 보정은 영상 잔차를 줄이면서 비현실적인 궤적을
  만들었다. 잔차 감소분 전체를 포즈 오차로 귀속하지 않는다.
  [거부된 포즈 출력](UW_SELFCAL_PREREG.md#result-both-dumps-are-refused-and-not-for-the-reason-expected)
- 세션별 강체 보정의 실패는 시간에 따른 내부 변형의 부재를 증명하지 않는다.
- 줄자 실험은 중요한 외부 길이 기준이다. 현재 누적 점군은 ARKit 포즈도 사용하므로,
  해당 국소 측정으로 모든 궤적·렌즈 주변부의 scale 정확도를 일반화하지 않는다.
- ARKit의 17 ms 내 포즈 오차 0.5–1 px는 외삽이다. 불명확한 readout 추정은
  검증된 6 ms 상한으로 취급하지 않는다. MultiCam에 ARKit 경로의 시간 모델을 전용하지 않는다.
- LiDAR를 입력받은 Pi3X 출력과 LiDAR 관측은 독립적인 두 증거가 아니다.
  CMDeviceMotion과 ARKit 역시 독립적인 raw 센서처럼 가정하지 않는다.
- tight walk의 높은 품질은 촬영 패턴과 장면이 함께 달라진 결과다.
  coverage 개수나 평균 영상 대비 점수만으로 인과적 결론을 내리지 않는다.

## 4. 목표 구조와 센서의 역할

```text
다양한 시점·화질의 멀티 카메라 RGB / LiDAR / 시간 / IMU
             ↓
visual·depth·IMU 기반 SLAM 관측 ↔ Pi3X 포즈·기하·confidence
             ↓
전체 window 정렬 + metric 제약
             ↓
공통 body 궤적·landmark 최적화
  RGB 재투영 + LiDAR 깊이 + rig + 재방문 + IMU 회전
             ↓
정확한 per-image metric pose + 융합 장면 기하
             ↓
3DGS 외관 재구성                 collision mesh 가능성 검증
  깊이 감독 + appearance/셔터      관측된 기하, 같은 scale·좌표계
             ↓
독립 metric / 정합 / 선명한 외관 / 충돌 형상 평가
```

| 정보 | 주 역할 | 제한 |
| --- | --- | --- |
| LiDAR | 실제 거리, scale, 관측된 표면의 기하 제약 | 평면 접선 방향 운동·미세 질감을 혼자 결정하지 못함 |
| wide RGB | 영상 정합과 안정적인 시각 초기화 | 저해상도 tracking 성능과 최종 렌더 질감은 다른 목표 |
| ultra-wide RGB | 주변부·추가 관측 방향·중앙부에서 바깥으로 이어지는 트랙 | 작은 stereo baseline만으로 정밀 원거리 깊이를 기대하지 않음 |
| IMU | 짧은 구간 회전, gravity, 시간축 보조 | 기존 처리된 신호를 raw IMU 사전적분 모델에 그대로 넣지 않음 |
| Pi3X | 포즈·형상 초기화와 불확실성을 가진 prior | 최종 metric 정답이나 보정 불가능한 고정 앵커로 두지 않음 |

두 렌즈의 카메라 포즈는 `T_world_camera(t) = T_world_body(t) T_body_camera`로
연결한다. 비동시 프레임은 실제 시각에서 공통 궤적을 평가한다.
초기에는 검증된 intrinsics·extrinsics를 고정하고, 보정 오차가 식별될 때만
제한된 자유도를 추가한다. 포즈·왜곡·노출·Gaussians를 처음부터 전부 자유롭게 풀지 않는다.

LiDAR 잔차는 원래 depth 격자와 좌표계에서 계산한다. 가림, 깊이 불연속,
유효 범위와 confidence를 반영하고, 깊이를 고해상도 RGB에 복제한 픽셀들을
독립적인 거리 측정으로 세지 않는다. 얇은 구조와 질감은 RGB 관측으로 보완한다.

## 5. 단계별 실행과 판정

### R1 — 비교 가능한 baseline과 평가 프로토콜

- 개발 데이터는 `d06152`, 재현 대조군은 `cb4586`을 우선 조사한다.
  `87bc2c`는 같은 공간의 다른 ARKit 촬영이지 같은 궤적의 ground truth가 아니다.
- split은 프레임 식별자로 고정하고, 필터나 pose 변경으로 다시 번호 매기지 않는다.
  렌즈·세션별 공통 영역 평가와 추가 영역 평가를 따로 만든다.
- 기존 `d06152` 실험의 wide 160×120 / ultra-wide 960×540을 재현 조건으로 기록한다.
  최종 외관 평가는 별도로 고정한 고해상도에서 한다.
- 각 단계의 수치 기준, 반복 횟수, 비용 상한과 중단 조건을 실행 전에 이 문서에 적는다.

완료 조건: 입력부터 평가까지 baseline을 설명·재현할 수 있고, 새 결과를
보고 나서 split이나 기준을 바꿀 필요가 없는 실행 명세가 있다.

### R2 — 순차 연결을 전역 window 정렬로 교체

- **W0:** 현재 `--join-scale-mode depth`.
- **W1:** 중복 프레임의 모든 위치·회전 추정을 보존하는 전역 정렬.
  window scale은 LiDAR 잔차로 제약하며, 이후 관측이 앞부분도 수정할 수 있게 한다.
- `free`는 과거 재현·실패 대조군, `median`은 scale 정책 비교용으로 남긴다.
- 알려진 scale perturbation, 회전 중심 이동, 순수 회전·짧은 baseline,
  누적 scale drift, 잘못된 overlap을 포함한 합성 통제를 먼저 통과한다.

판정: 최적화에 넣지 않은 트랙·깊이와 재방문 구간에서 개선되는가.
join 잔차만 줄어드는 것은 성공이 아니다. 실제 데이터가 지지하면 R3의 초기값으로 채택한다.

### R3 — RGB＋LiDAR＋rig의 metric bundle adjustment

- **W2:** W1에서 keyframe body 포즈와 공통 landmark를 함께 조정한다.
  두 렌즈 재투영, 관측 LiDAR 깊이, 검증된 재방문을 사용한다.
- LiDAR 중앙부에서 시작해 초광각 주변부로 이어지는 트랙을 활용한다.
  불확실한 depth anchor를 완벽한 고정 3D 점으로 취급하지 않는다.
- Pi3X는 초기값·완화 가능한 prior로 둔다. confidence와 overlap 잔차를
  바로 보정된 covariance라고 해석하지 않는다.
- 모델의 gauge를 명시하고 하나의 전역 좌표계를 고정한다. scale은 metric 관측으로
  제약하며, 오차를 숨기기 위한 구간별 Sim(3) 재정렬을 평가에 사용하지 않는다.

판정: W1/W2를 같은 입력과 평가에서 비교한다. metric 치수·독립 재투영을
개선하면서 고정 3DGS recipe의 외관이 유지·개선되는지 확인한다.
위치·회전이 비현실적으로 움직이거나 훈련 잔차만 좋아지면 추가 자유도를 채택하지 않는다.

### R4 — IMU 회전 제약, 이후 필요하면 관성 상태 확장

- **W3:** W2에 짧은 구간 상대 회전과 gravity 제약을 추가한다.
  카메라–IMU 회전과 시간 오프셋을 검증하고 불확실성을 반영한다.
- 정상 구간과 저질감·깊이 퇴화·빠른 회전 구간을 나눠 기여를 확인한다.
- 새 기록에서 raw gyro·accelerometer가 확보되면 바이어스·속도를 포함한
  사전적분을 별도 arm으로 검토한다. 이것이 R2/R3의 선행 조건은 아니다.

판정: 단순 포즈 보간과 W2보다 독립 회전·재투영·실패 구간에서 이득이 있는가.
관성 신호를 더했다는 이유만으로 기본 경로에 포함하지 않는다.

### R5 — geometry를 유지하면서 외관을 개선

- 같은 trajectory에서 기본 모델, 렌즈/세션별 저차원 광량·색 보정,
  노출 적분 모델을 차례로 비교한다. rolling shutter는 포맷별 보정 후 별도 추가한다.
- 초기에는 고정된 geometry로 원인을 분리한다. 이후 pose와의 상호작용이 의심되면
  `기존/보정 포즈 × 기본/개선 영상 모델`의 2×2 비교를 수행한다.
- 노출 중 움직임을 여러 시각의 렌더로 적분하는 모델은 선명한 평가 뷰에서 판단한다.
  흐린 입력을 더 잘 재현하는 것만으로 deblur 성공을 선언하지 않는다.
- 저블러 대조군에도 실제 blur가 있을 수 있다. 무블러 합성 대조군과 노출 0의
  구현 일치 검사를 구분하고, 효과는 binary 판정보다 blur 크기와 함께 보고한다.
- 고해상도 캐시가 부담이면 로딩·학습 방식을 바꾼다. 낮춘 해상도에서의 결과를
  최종 photorealism의 한계로 해석하지 않는다.

판정: 선명한 동일 타깃과 이동 렌더에서 외관을 개선하면서 독립 metric 기준을
악화시키지 않는가. pose·appearance·geometry 중 어느 것이 바뀌었는지 함께 기록한다.

### R6 — 적용 범위 검증

개발에 사용하지 않은 장면과 반복 촬영으로 넓힌다. 국소 detail 구간,
연결 구간, 재방문을 포함하고, 성공한 프레임뿐 아니라 실패 구간·복원 범위를 보고한다.
전 구간 평균이 괜찮아도 특정 구간이 metric 기준을 벗어나면 그 범위를 표시한다.

## 6. 평가에서 고정할 규칙

- 학습에 사용한 LiDAR와의 일치는 **센서 일관성**, 외부 길이·표면 기준과의
  일치는 **독립 정확도**로 구분한다. 한쪽 결과로 다른 쪽을 대체하지 않는다.
- 홀드아웃의 RGB·depth는 geometry 초기화, Pi3X conditioning, BA, 3DGS 학습에서
  제외한다. 기존 baseline이 이를 만족하는지 R1에서 감사하고, 아니라면 기존 재현 arm과
  누출을 제거한 arm을 구분한다.
- 평가 pose는 외부 타깃이나 별도의 localization 관측으로 결정한다.
  평가 RGB의 재구성 loss로 pose·appearance를 맞추지 않는다. localization에 사용한
  타깃/영역은 외관 평가에서 제외하고 그 범위를 기록한다.
- pose 최적화 전후를 비교할 때 같은 물리적 평가 카메라를 사용한다. 훈련 카메라만
  이동하고 평가 카메라가 다른 gauge에 남는 현상을 품질 저하로 오독하지 않는다.
- PSNR·SSIM·LPIPS는 같은 렌즈·해상도·타깃·mask에서 비교한다. 공통 영역의 품질 유지와
  추가 영역의 개선을 따로 보고, mean-image floor 대비 점수를 최종 품질의 주 지표로 삼지 않는다.
- 연속 프레임은 독립 표본으로 세지 않는다. paired 결과와 효과 크기를 유지하되,
  신뢰 구간은 촬영/장면 단위 또는 시간 블록의 상관을 반영한다.
- 동일 seed·budget 대조군을 먼저 실행하고, 개선 주장은 반복으로 확인한다.
  데이터 양이 달라지면 iteration·epoch·wall time을 함께 보고한다.
- 합성 주입 통제는 구현과 민감도를 검증한다. 실제 센서의 절대 정확도나
  nuisance 변수와의 분리까지 자동으로 증명하지는 않는다.

## 7. 필요한 촬영과 당장 확대하지 않을 작업

**C1 촬영 초안 — R1에서 포맷·기존 자료의 충족 여부를 확인해 구체화한다.**

- 같은 정적 실내에서 충분한 조명으로 멈춰 찍는 MultiCam 세트와 이동 촬영 세트.
  같은 표면을 여러 위치·높이에서 보고, 국소 관찰 구간 사이에 공통 landmark와 재방문을 둔다.
- 시작·중간·끝과 여러 방향에 외부 길이 기준을 배치한다. scale 보정용 기준과
  평가용 기준을 분리하고, 실제 길이 측정의 불확실성도 남긴다.
- 보정 타깃은 중앙과 코너, 여러 기울기에서 정지 촬영한다. 실제 활성 포맷의
  intrinsics·왜곡·rig를 검사하고, 움직이는 타깃 촬영은 시간/셔터 보정과 구분한다.
- 선명한 고정 평가 시점을 별도로 확보한다. 두 촬영 방식 사이의 장면·조명 변화를 기록한다.

R2/R3의 효과를 보기 전에는 ICP rate fusion 재튜닝, 새 pose foundation model로의
전면 교체, 대규모 3DGS densification sweep을 우선하지 않는다.
이미 보정된 초광각 영상에 공장 raw 왜곡 테이블을 다시 적용하지 않는다.
이 우선순위는 금지 규칙이 아니며, 새로운 근거가 생기면 변경 이력과 함께 수정한다.

## 8. 작업·실험 기록 방식

작업 시작 시 §1의 해당 행을 `진행`으로 바꾸고, 완료·중단 시 결과와 다음 작업을
같이 갱신한다. 진행 중인 실험은 명령·출력 위치·상태를 먼저 적어 재시작 때 중복 실행하지 않는다.
원본·렌더·checkpoint는 로컬에 두고, 이 문서에는 재현 명세와 결과 요약을 연결한다.

실험마다 다음 양식을 사용한다. 긴 실행 명세는 로컬 JSON/로그를 연결한다.

```text
실험 ID / 관련 단계 / 날짜 / 상태:
질문·가설:
대조군 / 바뀌는 변수:
입력·보정·split 식별자 / 코드·모델·환경 revision:
실행 명령 / seed / 해상도 / 비용 상한:
실행 전에 정한 판정·중단 기준:
출력·로그 위치:
결과: metric / 정합 / 외관 / 비용:
판정: 지지 / 반박 / 미확인:
한계·다음 행동:
```

연구 가설이 반박돼도 구현 검증은 통과할 수 있다. 기존 사전등록의 판정은 보존하고,
회귀 테스트와 연구 가설의 성립 여부를 구분해 기록한다.
새 작업에서 방향을 바꿀 때마다 별도의 계획서를 만들지 않고 이 문서를 갱신한다.

## 9. 진행·판단 변경 이력

| 날짜 | 항목 | 변경과 근거 | 다음 행동 |
| --- | --- | --- | --- |
| 2026-09-08 | R0 완료 | 구현·운영 이슈 중심의 리뷰에서, metric 제약·구간 내부 변형·영상 형성·독립 평가 중심으로 연구 목표를 재정렬했다. 기존 문서·융합 코드·저장된 MultiCam 렌더를 검토했으며 새 학습은 수행하지 않았다. | R1 baseline 및 평가 프로토콜 감사 |
| 2026-09-08 | 계획 수립 | 이 문서를 실행 계획과 progress의 단일 기준으로 만들었다. 기존 실험 기록은 수정하지 않고 근거로 연결했다. | R1 결과와 수치 판정 기준을 여기에 추가 |
| 2026-09-09 | R1 데이터 감사·부분 재현 | 80개 원본 세션, Pi3X 캐시 2개, 기존 checkpoint/export를 대조했다. 캐시의 기존 연결과 5개 RGB 렌더 arm을 CPU로 재채점했다. 두 렌즈 학습의 동일 wide 뷰 손해와 scale 고정만으로 남는 연결 실패를 확인했다. | R2 최소 실험의 기준 고정, 사라진 MultiCam export/split 복원 |
| 2026-09-09 | 기존 데이터의 활용 범위 확대 | 30 Hz RGB/depth ARKit와 노출 고정 MultiCam을 후보에 추가했다. 줄자·보드는 부분적으로 확보돼 있어 C1을 전면 미확보로 보지 않는다. 최종 독립 metric·sharp GT는 여전히 부족하다. | §1.1의 데이터별 실험을 진행하고 C1은 부족한 기준만 보충 |
| 2026-09-09 | R2-P1 완료·미채택 | 전역 window 정렬과 별도 재투영 평가를 구현·검증했다. 실패 세션에서 회전 순차 연결은 38.9%, 강건 전역 정렬은 23.3% 위치 차이를 줄였다. 전역화의 추가 이득이 없고 영상 pair가 1개씩뿐이라 채택 기준을 못 채웠다. 문제 window 자체의 강체 정렬 잔차 48.1 cm도 확인했다. | 짧은 간격의 평가 트랙 확보, 문제 구간의 국소 RGB/depth BA 준비. 기존 초기화 유지 |
| 2026-09-09 | 한 시간 R3/R4/R5 예비 검증 | 54 ARKit IMU 검사, 18 캐시 BA census, 3 MultiCam, 4개 시간 블록 반복, 4개 30 Hz 세션을 검사했다. 관측 연결성을 더하면 BA 예비 통과는 중앙 2구간뿐이고 시간 블록 반복에서는 0/4였다. prior·depth 보간 민감도도 결론을 바꾸지 않았다. | 관측 연결성·시차·조건별 오차를 먼저 다루고 frame-level holdout으로 GS 입력 재구성 |
| 2026-09-09 | O1/F1/F2/G1 완료 | 22구간 관측 민감도, 6세션 평가 사진 제외 Pi3X 재추론, 2세션×2포즈 GS 4000-step 대조를 완료했다. O1 통과 4/22, 사진 holdout 공동 기준 0/6, GS 외관 예비 기준 0/2였다. 깊이 일관성 개선과 외관 개선이 함께 나타나지 않았다. | 기존 줄자 독립 길이 검사, 회전/translation 제약별 ablation, 고정 기하에서 영상 형성 오차 분리. BA 기본 채택 보류 유지 |
| 2026-09-09 | 사용자가 전체 목표 재확인 | 멀티 카메라로 유효 영상을 최대한 확보하고 센서 기반 SLAM과 Pi3X를 융합해 per-image pose를 개선하며, PSNR·metric scale과 가능한 collision mesh를 함께 목표로 명시했다. | 전체 경로를 기준으로 실험 우선순위를 판단하고 R7 mesh 보조 목표 추가. 현재 BA 결과를 전체 융합 경로의 최종 판정으로 확대하지 않음 |
| 2026-09-16 | 최종 용도를 real2sim 스캐너로 확인, LiteReality-Agent 분석 반영 | 이전 판단: collision mesh 는 보조 목표, 순서는 R3 BA 판별 → 외관 분리 → 확대. 변경: 산출물은 시뮬레이터 환경(mesh + 3DGS, 한 좌표계)이고 같은 목표의 LiteReality-Agent 가 물리·의미 층을 이미 갖고 있어 R7 을 제3축으로 승격, "두 층이 우리 캡처에 붙는가"(S1~S3)를 1순위로. R7 은 BA 와 독립이라 R3 앞에 둔다. UW 는 RoomPlan·ARKit 층에 못 들어가므로 외관 층 전용으로 표기. | §1.6 표 순서대로: 커밋 → 포맷 변환기 → RoomPlan 공존 측정 → 저쪽 파이프라인 end-to-end → 줄자 길이 검사. R3 ablation·외관 분리는 유지하되 뒤로 |

## 10. 설계 참고

아래는 구조를 참고할 근거다. 이 프로젝트에서의 성능 향상을 입증하는 자료는 아니다.

- [Pi3X](https://github.com/yyfz/Pi3): 조건부 입력과 approximate metric scale.
- [MASt3R-SLAM](https://arxiv.org/abs/2412.12392): 학습된 국소 기하 prior와 전역 최적화의 결합.
- [FAST-LIVO2](https://arxiv.org/abs/2408.14035): 영상·LiDAR·IMU의 측정 수준 결합.
- [관성 사전적분](https://arxiv.org/abs/1512.02363): 바이어스·불확실성을 포함한 관성 제약.
- [BAD-Gaussians](https://arxiv.org/abs/2403.11831): 노출 중 궤적을 이용한 blur 형성 모델.
- [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/): 비선형 카메라와 rolling shutter를 지원하는 투영.
- [LiteReality-Agent](https://github.com/LiteReality) (로컬 `~/uv_workspace/LiteReality-Agent`):
  RoomPlan 껍질 + 생성 자산 + MuJoCo export. sim-ready 계약(`doc/Sim-Ready-intergration/Mujoco.md`),
  QC 게이트(`doc/QC/`), 뷰 선택·벽 스티칭(`doc/Tools/`), 캡처 포맷(`evidence_kit/read_scan.py`). §1.6.
