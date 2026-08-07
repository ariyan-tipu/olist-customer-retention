#!/usr/bin/env bash
# First-boot script for the scoring instance.
#
# Placeholders wrapped in @@ are filled in by bootstrap.py before the
# script is handed to EC2. Everything here is idempotent, so re-running
# it by hand on the instance is safe.
set -euxo pipefail

exec > >(tee /var/log/olist-bootstrap.log) 2>&1

# A micro instance has 1 GiB of memory, and pip resolving scientific
# wheels can exceed that. A gigabyte of swap costs nothing on the root
# volume and turns an out-of-memory kill into a slower install.
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends python3-venv python3-pip git nginx curl

id -u olist >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin olist
install -d -o olist -g olist /opt/olist /var/lib/olist

sudo -u olist python3 -m venv /opt/olist/venv
sudo -u olist /opt/olist/venv/bin/pip install --upgrade pip wheel
sudo -u olist /opt/olist/venv/bin/pip install "@@PIP_SPEC@@"

cat > /etc/olist.env <<EOF
OLIST_MODEL_PATH=/var/lib/olist/churn_model.joblib
OLIST_MODEL_S3_URI=@@MODEL_S3_URI@@
OLIST_CLOUDWATCH=1
AWS_REGION=@@REGION@@
EOF
chmod 0644 /etc/olist.env

cat > /etc/systemd/system/olist-scoring.service <<'EOF'
[Unit]
Description=Olist retention scoring service
After=network-online.target
Wants=network-online.target

[Service]
User=olist
Group=olist
EnvironmentFile=/etc/olist.env
ExecStart=/opt/olist/venv/bin/uvicorn --factory olist_retention.service.app:create_app \
  --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5
# The service reads one model out of S3 and writes nothing anywhere.
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
ReadWritePaths=/var/lib/olist

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/nginx/sites-available/olist <<'EOF'
server {
    listen 80 default_server;
    server_name _;

    # The scoring endpoint accepts batches, so the body limit is the one
    # setting worth naming here.
    client_max_body_size 2m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;
    }

    # Metrics and monitoring are readable by anyone who reaches the box,
    # which is the intent for a public demonstration. Restrict this block
    # by source address before putting anything private behind it.
    location /metrics {
        proxy_pass http://127.0.0.1:8000/metrics;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/olist /etc/nginx/sites-enabled/olist
nginx -t

systemctl daemon-reload
systemctl enable --now olist-scoring
systemctl restart nginx

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1/readyz >/dev/null 2>&1; then
    echo "scoring service is ready"
    exit 0
  fi
  sleep 5
done

echo "service did not become ready in time" >&2
systemctl status olist-scoring --no-pager || true
journalctl -u olist-scoring -n 100 --no-pager || true
exit 1
