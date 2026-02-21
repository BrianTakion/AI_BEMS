#!/usr/bin/env bash
set -euo pipefail

# Cron 작업 생성 (환경변수 포함)
cat > /etc/cron.d/analytics <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
TZ=${TZ:-Asia/Seoul}
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
POSTGRES_DB=${POSTGRES_DB:-appdb}
POSTGRES_USER=${POSTGRES_USER:-app}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-secret}

*/3 * * * * root /usr/local/bin/python /workspace/app/anomaly_detection.py >> /var/log/cron_anomaly.log 2>&1
*/3 * * * * root /usr/local/bin/python /workspace/app/model_control.py   >> /var/log/cron_control.log 2>&1
EOF

chmod 0644 /etc/cron.d/analytics

service cron start
service ssh start

# Jupyter 무인증 + 루트 허용 + 원격접속 허용
nohup jupyter notebook \
  --ServerApp.ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --notebook-dir=/workspace \
  --ServerApp.token='' --ServerApp.password='' \
  --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' \
  --NotebookApp.token='' --NotebookApp.password='' \
  >/var/log/jupyter.log 2>&1 &

echo "==================================="
echo "Container Started Successfully"
echo "==================================="
echo "SSH:"
echo "  - Internal Port: 22"
echo "  - External Port: 2222"
echo "  - Access: ssh root@localhost -p 2222"
echo "  - Password: dockerpass"
echo ""
echo "Jupyter Notebook:"
echo "  - Internal Port: 8888"
echo "  - External Port: 9999"
echo "  - Access: http://localhost:9999"
echo "  - Auth: No token / No password"
echo ""
echo "Cron Jobs: every 3 minutes"
echo "Timezone: ${TZ:-Asia/Seoul}"
echo "==================================="

# ★ 주피터 로그, 크론 로그도 함께 팔로우
#tail -F /var/log/jupyter.log /var/log/cron_anomaly.log /var/log/cron_control.log
tail -F /var/log/cron_anomaly.log /var/log/cron_control.log