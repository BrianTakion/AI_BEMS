FROM python:3.11-slim-bullseye

# --- 기본 환경 ---
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PIP_NO_CACHE_DIR=1

# --- 시스템 패키지 & 런타임 의존성 ---
# - libgomp1 : LightGBM OpenMP (Open Multi-Processing) 런타임
# - libfreetype6/libpng16-16 : Matplotlib 런타임
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    openssh-server \
    postgresql-client \
    tzdata \
    libgomp1 \
    libfreetype6 \
    libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

# --- SSH 설정(현 구성 유지; 운영 시 키 기반 권장) ---
RUN mkdir -p /var/run/sshd \
 && echo 'root:dockerpass' | chpasswd \
 && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# --- 작업 디렉토리 ---
WORKDIR /workspace

# --- Python 패키지 설치 ---
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- 애플리케이션 코드 ---
COPY app/ ./app/

# --- Cron 설정은 start.sh에서 동적으로 생성 (환경변수 포함) ---

# --- 로그 파일 생성 ---
RUN touch /var/log/cron_anomaly.log /var/log/cron_control.log

# --- 포트 노출 ---
EXPOSE 22 8888

# --- 시작 스크립트 ---
COPY start.sh /start.sh
RUN chmod +x /start.sh

# --- 헬스체크: cron/sshd 기동 확인 ---
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD pgrep -f "cron" > /dev/null && pgrep -f "sshd" > /dev/null || exit 1

CMD ["/start.sh"]

