#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$root/.cli-conda" ]]; then
  mapfile -t cli < "$root/.cli-conda"
  if [[ ! -x "${cli[0]:-}" || ! -d "${cli[1]:-}/conda-meta" ]]; then
    printf 'The Conda env recorded in %s is unavailable. Use the installed checkout or rerun its installer.\n' "$root/.cli-conda" >&2
    exit 1
  fi
  export PATH="$(dirname -- "${cli[0]}"):$PATH"
  # Start Conda without the caller's activation stack or its deactivation hooks.
  unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV
  exec "${cli[0]}" run --no-capture-output -p "${cli[1]}" device-jobs "$@"
fi
if [[ -x "$root/.venv/bin/device-jobs" ]]; then
  exec "$root/.venv/bin/device-jobs" "$@"
fi
python="${PYTHON:-python3}"
if ! "$python" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
  printf 'Device Jobs needs Python 3.10+. Set PYTHON or use the installed checkout.\n' >&2
  exit 1
fi
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m device_jobs.controller "$@"
