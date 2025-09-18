#!/bin/bash

# LDM-CRAM4 검증 테스트 스크립트
# 시뮬레이션을 실행하고 결과를 기준 데이터와 비교합니다.

set -e  # 에러 시 즉시 종료

echo "========================================"
echo "LDM-CRAM4 검증 테스트"
echo "========================================"

# 1. 시뮬레이션 컴파일
echo "1. 컴파일 중..."
if make clean && make; then
    echo "✓ 컴파일 성공"
else
    echo "❌ 컴파일 실패"
    exit 1
fi

# 2. 시뮬레이션 실행
echo "2. 시뮬레이션 실행 중..."
if timeout 300s ./ldm; then  # 5분 타임아웃
    echo "✓ 시뮬레이션 완료"
else
    echo "❌ 시뮬레이션 실패 또는 타임아웃"
    exit 1
fi

# 3. 결과 파일 존재 확인
RESULT_FILE="validation/concentration_grid_00720.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "✓ 결과 파일 생성됨: $RESULT_FILE"
else
    echo "❌ 결과 파일이 생성되지 않음: $RESULT_FILE"
    exit 1
fi

# 4. 검증 수행
echo "3. 결과 검증 중..."
if python validation_checker.py "$RESULT_FILE"; then
    echo ""
    echo "🎉 모든 검증 통과! 코드 변경사항이 올바릅니다."
    exit 0
else
    echo ""
    echo "⚠️  검증 실패! 코드를 확인하고 수정이 필요합니다."
    exit 1
fi