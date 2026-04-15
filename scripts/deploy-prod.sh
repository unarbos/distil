#!/bin/bash
set -euo pipefail

# Deploy the repo to the single-host distil runtime.
# Usage: ./scripts/deploy-prod.sh [api|frontend|validator|both|all]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REMOTE="${DISTIL_HOST:-distil}"
REMOTE_ROOT="${DISTIL_REMOTE_ROOT:-/opt/distil/repo}"
COMPONENT="${1:-all}"
LOCAL_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

case "$COMPONENT" in
    api|frontend|validator|both|all) ;;
    *)
        echo "Usage: $0 [api|frontend|validator|both|all]"
        exit 1
        ;;
esac

run_remote() {
    ssh "$REMOTE" "$1"
}

sync_repo() {
    echo "==> Syncing repo to $REMOTE:$REMOTE_ROOT..."
    run_remote "mkdir -p '$REMOTE_ROOT' '$REMOTE_ROOT/state'"
    rsync -az --delete --chown=distil:distil \
        --exclude='.git' \
        --exclude='.specstory' \
        --exclude='.pytest_cache' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='state/' \
        --exclude='frontend/.next/' \
        --exclude='frontend/node_modules/' \
        "$REPO_ROOT/" "$REMOTE:$REMOTE_ROOT/"
    run_remote "printf '%s\n' '$LOCAL_REV' > '$REMOTE_ROOT/REVISION' && chown distil:distil '$REMOTE_ROOT/REVISION'"
}

install_services() {
    echo "==> Installing distil systemd units on $REMOTE..."
    run_remote "bash '$REMOTE_ROOT/scripts/install_distil_services.sh'"
}

build_frontend() {
    echo "==> Building frontend on $REMOTE..."
    run_remote "cd '$REMOTE_ROOT/frontend' && npm install --no-fund --no-audit && NEXT_PUBLIC_API_URL=https://api.arbos.life npm run build"
}

restart_services() {
    echo "==> Restarting services: $*"
    run_remote "systemctl restart $*"
}

check_http() {
    local url="$1"
    local name="$2"
    local status
    status="$(run_remote "curl -s -o /dev/null -w '%{http_code}' '$url'")"
    if [ "$status" = "200" ]; then
        echo "==> $name healthy"
    else
        echo "==> WARNING: $name returned $status"
    fi
}

check_service() {
    local service="$1"
    if run_remote "systemctl is-active --quiet '$service'"; then
        echo "==> $service active"
    else
        echo "==> WARNING: $service is not active"
    fi
}

sync_repo
install_services

case "$COMPONENT" in
    api)
        restart_services distil-api
        check_http "http://localhost:3710/api/health" "API"
        ;;
    frontend)
        build_frontend
        restart_services distil-dashboard
        check_http "http://localhost:3720/" "dashboard"
        ;;
    validator)
        restart_services distil-validator
        check_service distil-validator
        ;;
    both)
        restart_services distil-api
        build_frontend
        restart_services distil-dashboard
        check_http "http://localhost:3710/api/health" "API"
        check_http "http://localhost:3720/" "dashboard"
        ;;
    all)
        build_frontend
        restart_services distil-api distil-dashboard distil-validator
        check_http "http://localhost:3710/api/health" "API"
        check_http "http://localhost:3720/" "dashboard"
        check_service distil-validator
        ;;
esac

echo "==> Deploy complete!"
