# 관측 가능성 → 사진 단위 제외 → GS 대조 실험

2026-09-09 06:33 KST 착수. [단일 계획서](../../docs/METRIC_RECONSTRUCTION_PLAN.md)의 §1.5에서 조건과 변경 이력을 관리한다.
기존 결과를 덮어쓰지 않고 별도 디렉터리에 실행했다. 이 폴더는 개인 세션에서 파생한 로컬 연구 산출물이다.

## R3-O1: 학습 관측만으로 검사한 안정성

이전 중앙 18개와 시간 블록 반복 4개를 모두 사용했다. 포즈 prior를 빼고 RGB/depth 측정의
Jacobian에서 landmark를 Schur 소거하여 translation 정보량을 계산했다.
회전은 기존 IMU에 고정하므로 결과는 **회전·센서 오차 모델을 조건으로 한 국소 민감도**다.
이를 보정된 confidence나 절대 metric 정확도로 해석하지 않는다.

training track을 정렬된 ID의 홀/짝으로 나누어 각각 추정한 위치도 비교했다.
원래 평가 track과 ARKit 위치는 이 계산·분할·최적화에 넣지 않았다.
전체/두 절반의 관측 연결, 조건부 std 최댓값 ≤2 cm, 두 결과의 이동 차이
중앙값 ≤1 cm/p95 ≤3 cm를 모두 만족한 것은 **4/22개**다:
`40ebb3`, `5b88f7`, `841e94` 중앙과 `40ebb3_first`.

- `cb4586`: 조건부 std 최댓값 1.37 cm지만 split 차이 중앙값 5.43 cm.
  수치적 정보량만으로 실제 최적화의 관측 민감도를 설명하지 못했다.
- `7d3d52`: 연결돼 있어도 조건부 std 최댓값 14.04 cm, split 중앙값 3.45 cm.
- 통과는 충분조건이 아니다. `841e94`와 `40ebb3_first`는 이전 ARKit 위치 비교에서 개선되지 않았다.
- 합성에서는 공통 depth에 5% scale bias를 주어도 완전한 rank와 작은 std,
  정확한 RGB/depth fit이 가능했다. **관측 가능성과 calibration 정확도는 다른 문제다.**

[사전등록](protocol.json), [집계](summary.json), [표](observations.csv), [개별 보고서](pilot/).

![관측 민감도와 제외 사진 평가](observability_and_holdout.png)

## R1-F1/R3-F1: 평가 사진을 Pi3X에서부터 제외

기존에 관측으로 연결됐던 중앙 7개에서 IMU 보정에 쓴 `cb4586`을 제외한 6개를 사용했다.
24장 중 index 3/9/15/21의 **사진과 depth를 Pi3X conditioning 및 BA에서 제외**했다.
학습 사진 20장만 임시 입력 디렉터리에 복사해 로컬 pinned Pi3X로 새로 추론했다.
모델에 pose는 입력하지 않았다. 입력은 training intrinsics와 confidence ≥2인 LiDAR이며,
scale은 training LiDAR와 predicted depth에서만 추정했다.

이제 [캐시](frame_holdout/inference/)에는 포즈뿐 아니라 dense predicted depth/confidence,
실제로 디코딩한 사진 목록·해시, tensor 크기, metric scale, 코드/weights revision이 있다.
기존 pose-only 캐시와 초기 조건이 다르므로 그 캐시 대비 개선 수치로 보고하지 않는다.
Pi3X source `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`, weights snapshot
`bb1deea4d7423de5b30691739cb451a3f57dc1d5`; GPU 추론 6개 모두 완료했다.

평가 카메라는 첫 training camera의 Pi3X↔ARKit rigid 변환 하나로 고정했다.
BA 전후에 같은 평가 camera를 쓰며 arm별 refit이나 scale fitting이 없다.
ARKit은 독립 metric GT가 아니고, 이번 구간 선택에는 이전 전체 영상 결과를 사용했으므로
**개발 세트의 frame holdout**이다. 보지 않은 세션의 확증 검증으로 해석하지 않는다.

F1은 source 특징점도 training track에서 12 px 이상 떨어진 것만 사용했다.
평가 pair ≥10·camera ≥3, RGB ≥15% 개선·depth abs-rel 악화 ≤10%를 모두 통과한 것은 0/6개다.
표본이 충분한 `5b88f7`은 18.26→22.94 px로 악화했고, `841e94`는 7.06→6.88 px였다.

F2는 training landmark를 완전히 제외된 **새 target 사진**에 투영하는 평가도 유효하므로,
source 거리 배제만 없앴다. pair 제안/매칭/깊이/판정 기준은 유지했고 F1을 덮어쓰지 않았다.
이 조건에서도 전체 기준 통과는 0/6개다.

| 세션 | F2 pair / 평가 사진 | Pi3X → BA 재투영 px | Pi3X → BA depth abs-rel |
| --- | --- | --- | --- |
| 40ebb3 | 15 / 4 | 19.92 → 5.35 | 0.0088 → 0.0102 |
| 5b88f7 | 12 / 4 | 16.71 → 22.67 | 0.0097 → 0.0146 |
| 5bd1ed | 8 / 3 | 5.96 → 5.95 | 0.0082 → 0.0078 |
| 683ef1 | 2 / 2 | 5.65 → 13.09 | 0.0055 → 0.0215 |
| 7d3d52 | 0 / 0 | 평가 불가 | 평가 불가 |
| 841e94 | 15 / 4 | 6.89 → 6.86 | 0.0063 → 0.0069 |

[정확한 분할·F1 조건](frame_holdout/protocol.json), [F2 변경 이유](frame_holdout/protocol_source_all.json).
각 [trial](frame_holdout/trials/)에 명령·로그·새 트랙·BA·training screen·평가 관측을 남겼다.

## R3-G1: 3DGS 포즈 대조

F1에서 충분한 pair와 4개 평가 사진을 확보한 두 세션 `5b88f7`, `841e94`를 모두 사용한다.
F2 결과를 보고 GS 세션을 바꾸지 않았다. 20 training / 4 evaluation 사진, seed 42,
DN-Splatter 4000 step, 960×720, camera optimization off, depth L1 λ0.2,
depth-normal λ0.05를 고정했다. 각 세션의 Pi3X/BA arm은 **동일한 초기 Gaussian 점군**을
사용하며 그 점군에는 evaluation depth가 없다. 변하는 것은 training camera pose뿐이다.

원래 num_downscales=2 조건은 DN-Splatter의 depth-normal 격자 오류로 학습 step 완료 전에
실패했다. 외부 코드를 수정하지 않고 모든 arm을 num_downscales=0으로 바꿨다.
[원래 명세](gs_pilot/protocol.json), [수정 명세](gs_pilot/protocol_fullres.json), 실패 로그를 보존했다.
기존 7000-step 연구와는 다른 단기 대조이며 동일 세션/동일 예산의 두 arm만 비교한다.

[export 감사](gs_pilot/export_reports.json)는 사진·depth·평가 camera·초기 PLY가 두 arm에서
같음을 확인한다. parser 로그도 training 20개와 evaluation index 3/9/15/21을 확인했다.
학습 중 평가를 끄고 4000 step의 마지막 checkpoint만 채점하므로 평가 점수로 checkpoint를
선택하지 않는다. 시점별 float 렌더·PSNR/SSIM/LPIPS·depth 일관성을 별도로 저장한다.

주의할 평가 구현 차이가 있었다. DN의 aggregate evaluator는 수치를 계산한 뒤 train mode로
돌아가 저장용 렌더를 만든다. 따라서 별도 `score_gs_fixed_views.py`는 모델을 eval mode에
유지하고 같은 forward 결과에서 수치와 이미지를 저장하며, 저장한 float 배열로 PSNR을 대조한다.
상류 코드의 절대 경로 처리로 생성된 중복 디렉터리도 기록을 남겨 지정한 산출물 경로로 이동한다.

네 개 학습 모두 step 3999까지 완료했고 같은 checkpoint를 고정 평가했다.
평가 스크립트의 첫 실행은 venv/bin이 PATH에 없어 Ninja를 찾지 못했다. 학습과 같은
PATH를 적용해 재실행했으며 실패 로그·부분 출력도 보존했다. 모델/학습 설정은 바꾸지 않았다.
최종 시점별 결과는 [fixed_view_scores_retry](gs_pilot/fixed_view_scores_retry/)에 있다.

| 세션 | Pi3X → BA PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | LiDAR RMSE cm ↓ | depth abs-rel ↓ |
| --- | --- | --- | --- | --- | --- |
| 5b88f7 | 16.54 → 16.29 | 0.611 → 0.599 | 0.332 → 0.334 | 5.98 → 4.72 | 0.0402 → 0.0387 |
| 841e94 | 17.45 → 17.27 | 0.616 → 0.615 | 0.202 → 0.190 | 2.24 → 2.06 | 0.0120 → 0.0146 |

각 값은 평가 사진 4장의 metric 평균이다. 두 세션 모두 PSNR 개선은 2/4장으로,
사전등록한 ≥3/4장 및 평균 +0.5 dB 기준을 충족하지 못했다. LiDAR depth는 confidence ≥2,
유효 depth >0.1 m에서 비교하며 scale fitting이 없다. 저장한 float 배열의 RGB PSNR과
depth RMSE/abs-rel이 원래 수치를 재현하고, 상류 aggregate 평가와도 1e-5 이내로 일치했다.
사진의 해시와 평가 camera 행렬도 두 arm에서 동일함을 검증했다.
[집계·시점별 차이](gs_pilot/summary.json), [검증 및 그림 생성 코드](summarize_gs.py).

![모든 평가 시점의 paired metric](gs_pilot/paired_metrics.png)

전체 평가 사진을 빠짐없이 비교했다. `5b88f7`에서는 책장 모서리·글자·반사 영역의
번짐과 변형이 남고, `841e94`에서도 모니터 글자와 선반/벽의 형태·질감 불일치가 보인다.
원본 자체의 blur도 있어 선명한 정답 대비 품질과 같지 않다. 특정 뷰를 고르지 않고
[5b88f7 전체](gs_pilot/5b88f7_all_views.png), [841e94 전체](gs_pilot/841e94_all_views.png)를 저장했다.

이 단기 대조에서 현재 BA 조합의 기하·외관 공동 개선 근거는 얻지 못했다.
포즈 교체에는 회전 고정과 translation 최적화가 함께 포함돼 있어 어느 제약이 원인인지
확정할 수 없다. 하나의 seed와 4개 시점으로 작은 차이의 통계적 일반성을 주장하지 않는다.
공통 점군은 Pi3X pose로 만들었기 때문에 초기 정합에서 Pi3X arm에 유리할 수 있다.
따라서 이 결과는 동일 점군에서 pose를 교체한 효과이며 BA로 점군까지 재생성한
전체 파이프라인의 판정은 아니다. 기하 검증이 지지되면 pose/점군의 2×2 대조로 분리한다.
다음은 기존 줄자의 독립 길이 검사, 새로운 시간 블록의 제약별 ablation,
기하를 고정한 노출/blur·반사 영역 분석이다. 큰 GS sweep으로 바로 확대하지 않는다.

## 검증과 재현

```bash
OPENBLAS_NUM_THREADS=1 python3 eval/test_translation_information.py
python3 eval/test_frame_holdout.py
python3 eval/test_local_rgbd_ba.py
python3 tools/test_export_3dgs.py
python3 tools/read_session.py /tmp/nav-r2-p1-fixture-20260909/20260807-014530-fixture
```

[검증 로그](checks/)에는 analytic Jacobian/dense Schur 일치, gauge/nullspace,
held-out 관측 변경의 무영향, 공통 scale bias의 비검출, 평가 이미지/depth 미접근,
동일 평가 gauge 및 위치 변화 민감도를 남겼다.
원본 세션·외부 모델 checkout은 수정하지 않았다. 원래 연구의 LiDAR/ARKit 상관과
선명한 독립 정답의 부족은 해결된 것으로 간주하지 않는다.

[검사 결과](checks/results.json), [코드 snapshot](code/), [실행 환경·revision](provenance.json),
[원본 센서 해시](sensor_input_manifest.json), [전체 산출물 해시](artifact_manifest.json)를 보존한다.
Pi3X/GS 패키지 목록은 `pi3_environment.txt` / `gs_environment.txt`에 있다.
직전 실험의 입력·IMU 보정은 이전 artifact manifest의 해시로 연결했고, 이번 여섯 원본
세션의 센서 파일을 다시 해시해 이전 기록과 같은지도 확인했다. 산출물은 private local 데이터다.
