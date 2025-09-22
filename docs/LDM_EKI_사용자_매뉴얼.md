# LDM-EKI 시스템 사용자 매뉴얼

작성일: 2025년 9월 22일  
버전: 1.0  
대상: 연구자 및 개발자  

이 매뉴얼은 LDM-EKI (라그랑지안 확산 모델과 앙상블 칼만 역산) 시스템의 설치, 설정, 실행에 대한 실무 가이드입니다.

시스템 개요

LDM-EKI 시스템은 대기 중 방사능 물질 확산을 시뮬레이션하고 역추정하는 통합 플랫폼입니다. GPU 가속 입자 추적과 베이지안 추론을 결합하여 소스항을 추정합니다.

주요 구성요소
- LDM (라그랑지안 확산 모델): CUDA 기반 입자 추적 시뮬레이션
- EKI (앙상블 칼만 역산): Python 기반 베이지안 소스항 추정
- 통합 워크플로우: 자동화된 데이터 교환 및 피드백 루프

시스템 요구사항

하드웨어 요구사항
- NVIDIA GPU (Compute Capability 8.0 이상 권장)
- 메모리: 최소 16GB RAM, 권장 32GB RAM
- GPU 메모리: 최소 8GB VRAM
- 저장공간: 최소 50GB 여유 공간

소프트웨어 요구사항
- Ubuntu 20.04 이상 또는 CentOS 8 이상
- CUDA Toolkit 11.8 이상
- GCC 9.0 이상
- Python 3.8 이상
- OpenMPI 4.0 이상

설치 및 설정

1. 저장소 클론

git clone https://github.com/username/LDM-EKI.git
cd LDM-EKI

2. 환경 설정

환경 변수 설정:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MPI 환경 확인:
mpirun --version

3. 컴파일

LDM 시뮬레이터 컴파일:
cd ldm
make clean
make

실행 파일 확인:
ls -la ldm_main_safe ldm_sequential

4. Python 환경 설정

필요한 패키지 설치:
pip install numpy scipy matplotlib pandas

EKI 모듈 확인:
cd eki/src
python -c "import numpy; print('NumPy 설치 확인됨')"

기본 실행 방법

1. 단일 LDM 시뮬레이션

기본 시뮬레이션 실행:
cd ldm
./ldm_main_safe

시뮬레이션 설정은 data/input/setting.txt에서 수정할 수 있습니다.

출력 확인:
ls output/
ls logs/ldm_logs/

2. 앙상블 LDM 시뮬레이션

앙상블 시뮬레이션 실행:
./ldm_sequential

앙상블 설정:
- 앙상블 수: 100개 (기본값)
- 앙상블당 입자수: 1,000개
- 총 입자수: 100,000개

출력 확인:
ls output_ens/
ls logs/integration_logs/

3. EKI 추정

EKI 실행:
cd eki/src
python RunEstimator.py

설정 파일: config/input_config에서 파라미터 조정 가능

결과 확인:
ls results/

설정 파일

1. LDM 설정 (ldm/data/input/setting.txt)

주요 파라미터:
- NOP: 입자 수 (기본값: 100000)
- TIME_TOTAL: 총 시뮬레이션 시간 (초)
- TIME_STEP: 시간 간격 (초)
- SOURCE_X, SOURCE_Y, SOURCE_Z: 소스 위치

2. EKI 설정 (eki/config/input_config)

주요 파라미터:
- ensemble_size: 앙상블 수 (기본값: 100)
- time_intervals: 시간 구간 수 (기본값: 24)
- receptor_count: 리셉터 수 (기본값: 3)

출력 파일 구조

LDM 출력
logs/ldm_logs/
├── particles_final.csv         # 최종 입자 위치
├── particles_15min_*.csv       # 시간별 입자 상태
└── simulation.log              # 시뮬레이션 로그

output/
└── plot_*.vtk                  # 시각화 파일 (ParaView 호환)

EKI 출력
logs/eki_logs/
├── ensemble_observations_iter_*.csv    # 관측 매트릭스
├── estimated_sources_iter_*.csv        # 추정된 소스항
└── convergence_log.txt                 # 수렴 로그

eki/results/
├── plot_receptor_*_*.csv      # 리셉터별 결과
└── plot_receptor_*_*.png      # 시각화 그래프

문제 해결

1. CUDA 관련 오류

오류: "CUDA driver version is insufficient"
해결: NVIDIA 드라이버 업데이트
sudo apt update && sudo apt install nvidia-driver-latest

오류: "nvcc not found"  
해결: CUDA PATH 설정 확인
which nvcc
export PATH=/usr/local/cuda/bin:$PATH

2. 메모리 관련 오류

오류: "out of memory"
해결: 입자 수 또는 앙상블 수 감소
- setting.txt에서 NOP 값 줄이기
- input_config에서 ensemble_size 줄이기

3. 실행 권한 오류

오류: "Permission denied"
해결: 실행 권한 설정
chmod +x ldm_main_safe ldm_sequential

4. Python 모듈 오류

오류: "ModuleNotFoundError"
해결: 필요 모듈 설치
pip install -r requirements.txt

고급 사용법

1. 사용자 정의 소스 설정

다중 소스 설정:
setting.txt에서 소스 위치 추가

시간 변화 방출률:
EKI 설정에서 시간별 방출률 패턴 정의

2. 리셉터 위치 설정

리셉터 좌표 수정:
eki/config/input_data에서 리셉터 위치 설정

3. 시각화

ParaView를 사용한 3D 시각화:
paraview output/plot_*.vtk

Python을 사용한 결과 분석:
python ldm/simple_visualize.py
python eki/src/plot_results.py

성능 최적화

1. GPU 최적화

최적의 블록 크기 설정:
소스 코드에서 BLOCK_SIZE 조정

메모리 사용량 모니터링:
nvidia-smi

2. CPU 최적화

OpenMP 스레드 수 설정:
export OMP_NUM_THREADS=8

3. I/O 최적화

SSD 사용 권장
출력 파일을 고속 디스크에 저장

모니터링 및 로깅

1. 실시간 모니터링

GPU 사용률:
watch -n 1 nvidia-smi

메모리 사용량:
top -u $USER

2. 로그 분석

시뮬레이션 진행상황:
tail -f logs/ldm_logs/simulation.log

EKI 수렴성:
tail -f logs/eki_logs/convergence_log.txt

자주 묻는 질문

Q: 시뮬레이션이 너무 오래 걸립니다.
A: 입자 수를 줄이거나 더 강력한 GPU를 사용하세요. 또한 시간 간격을 늘려보세요.

Q: 결과가 수렴하지 않습니다.
A: EKI 파라미터를 조정하거나 앙상블 수를 늘려보세요. 관측 데이터의 품질도 확인하세요.

Q: 메모리 오류가 발생합니다.
A: 입자 수를 줄이거나 시스템 메모리를 늘리세요. 배치 처리를 고려해보세요.

Q: 컴파일 오류가 발생합니다.
A: CUDA 버전과 GCC 버전 호환성을 확인하세요. Makefile의 설정을 점검하세요.

지원 및 연락처

기술 지원: jrpark@example.com
문서 업데이트: https://github.com/username/LDM-EKI/docs
이슈 보고: https://github.com/username/LDM-EKI/issues

참고 문헌

1. LDM 모델: "라그랑지안 입자 확산 모델링" (2023)
2. EKI 방법론: "앙상블 칼만 역산법" (2022)
3. CUDA 프로그래밍: "GPU 가속 컴퓨팅" (2024)

부록

A. 설정 파일 예제
B. 에러 코드 목록  
C. 성능 벤치마크
D. API 참조

버전 히스토리

1.0 (2025-09-22): 초기 버전 릴리즈
- 기본 LDM-EKI 통합 기능
- 앙상블 시뮬레이션 지원
- 사용자 매뉴얼 완성