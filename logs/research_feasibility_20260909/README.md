# 확보 데이터의 연구 실험 가용성 감사

날짜: 2026-09-09 (Asia/Seoul). 계획과 판단의 기준은
[METRIC_RECONSTRUCTION_PLAN.md §1.1](../../docs/METRIC_RECONSTRUCTION_PLAN.md#11-확보-데이터로-검증-가능한-범위--2026-09-09-감사)다.
이 디렉터리는 로컬 감사 산출물이며 원본 사진·센서 binary·checkpoint를 포함하지 않는다.

저장소 루트에서 실행:

```bash
python3 logs/research_feasibility_20260909/audit.py
python3 logs/research_feasibility_20260909/cache_probe.py
python3 logs/research_feasibility_20260909/rescore.py
```

Python, NumPy, Pillow를 사용한다. GPU·네트워크·모델 추론·3DGS 재학습은 사용하지 않았다.
스크립트는 입력을 읽고 같은 디렉터리의 JSON 보고서를 갱신한다.

| 출력 | 검사 범위 | 결과 |
| --- | --- | --- |
| `inventory.json` | `/home/myeongcheol/nav_data`의 manifest, RGB/depth/motion/pose 인덱스, 사진 존재·기록 byte 수, depth/confidence 파일 끝 위치 | 80개: ARKit 54/MultiCam 26; 두 렌즈 22; 활성 depth K가 있는 MultiCam 17; 모두 depth/motion 있음. 검사한 수량·파일 범위 불일치 없음 |
| `cache_probe.json` | `pi3win/2994fa.npz`, `7d3d52.npz` 및 `pi3traj` 참조. 기존 first-estimate-wins 위치 기반 연결을 free/scale=1로 재생성 | ARKit 대비 전역 SE(3) 정렬 후 평균 위치 차이: 2994fa 2.452/2.463 m, 7d3d52 0.261/0.244 m. RMSE는 별도 필드 |
| `rescore.json` | `gs3d/runs/new_d06152`, `new_87bc2c` 저장 PNG의 RGB PSNR | 문서의 중앙값 재현. 같은 wide GT 35장의 mixed−source 평균 −0.764 dB, 중앙값 −0.705 dB |
| `provenance.json` | 이번 감사의 도구/입력/학습 config 식별 정보 | 원본 Pi3X 추론 당시 모델 revision을 복구한 기록은 아님 |

`inventory.json`의 timestamp 통계는 각 스트림의 기록 순서 기준이다.
`nearest_depth`와 `wide_uw_nearest`는 저장된 관측 사이의 최근접 시간 거리이며,
센서 clock offset의 추정치가 아니다. `issues: {}`는 timestamp나 센서 품질까지
문제가 없다는 의미가 아니다. 중복/역순 timestamp는 스트림별 `nonincreasing`에 남는다.
위치·heading·장치 이름은 이 보고서에 수집하지 않았다.

`rescore.json`은 저장 PNG를 리사이즈 없이 비교한 결과다. DN JSON의 RGB 평균과
PNG 재채점 평균은 양자화 등의 영향으로 소폭 다르며, 내부 evaluator를 재실행한 것은 아니다.
SSIM·깊이 오차를 이번에 재현하지 않았다. 원본 export가 사라진 arm은 렌더가 남아도
입력·split을 완전히 재현했다고 판정하지 않는다.

`cache_probe.py`는 새로운 전역 window optimizer가 아니다. 회전은 기존 위치 fit으로
변환한 뒤 잔차만 측정했다. 참조 ARKit 포즈는 최적화에 사용하지 않고 평가 정렬에만 사용한다.
두 캐시가 당시 어떤 conditioning 인자와 모델 revision으로 생성됐는지는 파일만으로 확정할 수 없다.
