# 가벼운 파이썬 베이스 이미지
FROM python:3.12-slim

WORKDIR /app

# 1) 의존성 먼저 설치
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# 2) 코드 + 저장된 모델 + 샘플 입력 복사
COPY src/ ./src/
COPY models/ ./models/
COPY data/sample_input.csv ./data/sample_input.csv

# 3) 추론 엔트리포인트
ENTRYPOINT ["python", "src/inference.py"]
CMD ["data/sample_input.csv"]