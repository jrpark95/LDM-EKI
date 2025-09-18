# LDM-EKI 통합 시스템

LDM (Lagrangian Dispersion Model)과 EKI (Ensemble Kalman Inversion)를 통합한 역산 시스템입니다.

## 구조

```
LDM-EKI/
├── ldm/           # LDM 코어 시스템 (CRAM7 기반)
├── eki/           # EKI 최적화 시스템
├── integration/   # 통합 인터페이스
├── data/          # 입출력 데이터
├── scripts/       # 유틸리티 스크립트
└── docs/          # 문서화
```

## 사용법

### 1. 빌드
```bash
cd ldm && make
```

### 2. 실행
```bash
# LDM 서버 시작
cd ldm && ./ldm

# EKI 클라이언트 실행 (다른 터미널에서)
cd eki && python src/RunEstimator.py config/input_config config/input_data
```

## 주요 기능

- **LDM**: GPU 기반 라그랑지안 분산 모델링
- **EKI**: 앙상블 칼만 역산을 통한 소스 추정
- **함수 트래킹**: EKI 시스템의 함수 사용량 모니터링
- **통합 워크플로우**: LDM-EKI 자동화된 커플링

## 개발자 정보

- LDM: LDM-CRAM7 기반
- EKI: eki-20241030 기반
- 통합: 2025년 9월