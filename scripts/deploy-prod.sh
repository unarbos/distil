#!/bin/bash
# Deploy API + Frontend to production server (distil-api / 46.224.105.143)
# Usage: ./scripts/deploy-prod.sh [api|frontend|both]
set -euo pipefail

REMOTE="distil-api"
COMPONENT="${1:-both}"

deploy_api() {
    echo "==> Deploying API to $REMOTE..."
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
        /home/openclaw/distillation/api/ "$REMOTE:/opt/distil/api/"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
        /home/openclaw/distillation/eval/ "$REMOTE:/opt/distil/eval/"
    echo "==> Restarting distil-api service..."
    ssh -t "$REMOTE" 'systemctl restart distil-api'
    sleep 2
    STATUS=$(ssh "$REMOTE" 'curl -s -o /dev/null -w "%{http_code}" http://localhost:3710/api/health')
    if [ "$STATUS" = "200" ]; then
        echo "==> API deployed and healthy ✓"
    else
        echo "==> WARNING: API returned $STATUS after deploy!"
    fi
}

deploy_frontend() {
    echo "==> Deploying frontend to $REMOTE..."
    rsync -avz --exclude='node_modules' --exclude='.next' --exclude='.git' \
        /home/openclaw/distillation-frontend/ "$REMOTE:/opt/distil/frontend/"
    echo "==> Building frontend on $REMOTE..."
    ssh -t "$REMOTE" 'cd /opt/distil/frontend && NEXT_PUBLIC_API_URL=https://api.arbos.life npx next build'
    echo "==> Restarting distil-dashboard service..."
    ssh -t "$REMOTE" 'systemctl restart distil-dashboard'
    sleep 2
    STATUS=$(ssh "$REMOTE" 'curl -s -o /dev/null -w "%{http_code}" http://localhost:3720/')
    if [ "$STATUS" = "200" ]; then
        echo "==> Frontend deployed and healthy ✓"
    else
        echo "==> WARNING: Frontend returned $STATUS after deploy!"
    fi
}

case "$COMPONENT" in
    api)      deploy_api ;;
    frontend) deploy_frontend ;;
    both)     deploy_api; deploy_frontend ;;
    *)        echo "Usage: $0 [api|frontend|both]"; exit 1 ;;
esac

echo "==> Deploy complete!"
