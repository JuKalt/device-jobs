"""Standalone remote helper, sent over SSH; requires only Python's standard library."""
from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def group_members(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return members



def belongs_to_attempt(pid, directories):
    try:
        entries = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
        return any(('VCI_REMOTE_ATTEMPT_DIR=' + str(d)).encode() in entries for d in directories)
    except OSError:
        return False


def attempt_members(directory):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and belongs_to_attempt(int(entry.name), [directory]):
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


TERMINAL_STATES = ('succeeded', 'failed', 'cancelled')


def boot_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return None


def process_start(pid):
    """Kernel start time of a process; distinguishes a live process from PID reuse."""
    try:
        return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def supervisor_state(directory):
    """alive, dead (rebooted, killed or exited) or missing (never launched)."""
    try:
        record = json.loads((Path(directory) / 'supervisor.json').read_text())
    except FileNotFoundError:
        return 'missing'
    except (OSError, ValueError):
        return 'missing'
    if record.get('boot_id') != boot_id():
        return 'dead'
    start = process_start(record.get('pid'))
    return 'alive' if start is not None and start == record.get('start') else 'dead'


def stop_attempt(pgid, directory, grace=5):
    # MAMMA launches new sessions internally, so process-group cleanup alone
    # would miss its workers. The inherited attempt marker also identifies them.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + grace
        while True:
            if pgid is not None:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
            for pid in attempt_members(directory):
                try:
                    if belongs_to_attempt(pid, [directory]):
                        os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if (pgid is None or not group_members(pgid)) and not attempt_members(directory):
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
    return False


def query(arguments):
    result = subprocess.run(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True)
            if row]


def number(value):
    """nvidia-smi reports unsupported fields as [N/A] or [Not Supported]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def probe(attempts):
    own_groups = set()
    for directory in attempts:
        try:
            state = json.loads((Path(directory) / 'status.json').read_text())
            if state['state'] == 'running':
                own_groups.add(int(state['pgid']))
        except (OSError, ValueError, KeyError):
            pass
    # nvidia-smi queries are not an atomic snapshot. Bracket the process sample
    # so an allocation appearing (or disappearing) between queries cannot alone
    # make our own memory look like external work.
    gpu_query = ['--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu']
    before = {row[1]: number(row[3]) for row in query(gpu_query) if len(row) == 5}
    own_memory = {}
    unmeasured = set()
    for row in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']):
        if len(row) != 3:
            continue
        uuid, pid, used = row
        try:
            if os.getpgid(int(pid)) in own_groups or belongs_to_attempt(int(pid), attempts):
                if number(used) is None:
                    unmeasured.add(uuid)
                else:
                    own_memory[uuid] = own_memory.get(uuid, 0) + number(used)
        except (ProcessLookupError, ValueError):
            pass
    devices = {}
    for row in query(gpu_query):
        if len(row) != 5:
            continue
        index, uuid, total, used, utilization = row
        total, used = number(total), number(used)
        value = {'uuid': uuid, 'total_mib': total, 'used_mib': used, 'utilization': number(utilization)}
        if total and used is not None and before.get(uuid) is not None and uuid not in unmeasured:
            # Includes graphics processes and unidentifiable allocations conservatively.
            foreign = max(0., min(before[uuid], used) - own_memory.get(uuid, 0.))
            value.update(foreign_fraction=foreign / total, foreign_mib=foreign)
        else:
            # Unknown occupancy: the controller neither launches here nor cancels jobs.
            value.update(foreign_fraction=None, foreign_mib=None)
        devices[index] = value
        devices[uuid] = value
    return devices


def machine_probe(attempts):
    """GPU occupancy plus liveness of this machine's attempt supervisors."""
    supervisors = {directory: supervisor_state(directory) for directory in attempts}
    try:
        return dict(devices=probe(attempts), supervisors=supervisors, gpu_error=None)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # The machine is reachable; a failed GPU query must not expire leases.
        return dict(devices={}, supervisors=supervisors, gpu_error=f'nvidia-smi query failed: {exc}')


def reap(directory):
    """Stop the leftovers of an attempt whose supervisor is gone, then finalize its status."""
    directory = Path(directory)
    state = supervisor_state(directory)
    if state == 'alive':
        return dict(state='alive')
    try:
        status = json.loads((directory / 'status.json').read_text())
    except (OSError, ValueError):
        status = {}
    if status.get('state') in TERMINAL_STATES:
        return dict(state=status['state'])
    (directory / 'cancel').touch()
    # The recorded process group may belong to a previous boot or be reused;
    # the inherited attempt marker identifies the job's processes safely.
    if not stop_attempt(None, directory):
        write_json(directory / 'status.json', {'state': 'blocked',
                                               'reason': 'attempt workers still alive after supervisor loss'})
        return dict(state='blocked')
    reason = ('supervisor never started' if state == 'missing'
              else 'supervisor lost' if status.get('state') != 'blocked' else 'blocked workers stopped')
    write_json(directory / 'status.json', {'state': 'cancelled', 'reason': reason})
    return dict(state='cancelled')


def supervise(payload):
    directory = Path(payload['attempt_dir'])
    process = None
    try:
        if time.time() - (directory / 'heartbeat').stat().st_mtime > payload['lease_seconds']:
            write_json(directory / 'status.json', {'state': 'cancelled', 'reason': 'expired before launch'})
            return
        if (directory / 'cancel').exists():
            write_json(directory / 'status.json', {'state': 'cancelled', 'reason': 'cancelled before launch'})
            return
        env = dict(os.environ, **payload['env'])
        env['VCI_REMOTE_ATTEMPT_DIR'] = str(directory)
        with (directory / 'job.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(payload['command'], cwd=payload['cwd'], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write_json(directory / 'status.json', {'state': 'running', 'pgid': process.pid})
            reason = None
            started = time.monotonic()
            while process.poll() is None:
                if payload.get('timeout_seconds') and time.monotonic() - started > payload['timeout_seconds']:
                    reason = 'command timeout'
                    break
                if (directory / 'cancel').exists():
                    reason = 'GPU unavailable or controller stopped'
                    break
                if time.time() - (directory / 'heartbeat').stat().st_mtime > payload['lease_seconds']:
                    reason = 'controller lease expired'
                    break
                time.sleep(.25)
            # Clean up child workers even if the main command exits first.
            if not stop_attempt(process.pid, directory):
                write_json(directory / 'status.json', {'state': 'blocked', 'pgid': process.pid,
                                                       'reason': 'attempt workers still alive'})
                return
            process.wait()
            write_json(directory / 'status.json', {
                'state': 'cancelled' if reason else ('succeeded' if process.returncode == 0 else 'failed'),
                'returncode': process.returncode, 'reason': reason})
    except BaseException as exc:
        if process is not None and not stop_attempt(process.pid, directory):
            write_json(directory / 'status.json', {'state': 'blocked', 'reason': str(exc)})
        else:
            write_json(directory / 'status.json', {'state': 'failed', 'reason': str(exc)})



def provision(payload):
    """Execute configured check/install/check steps inside the supervised group."""
    directory = Path(payload['attempt_dir'])
    env = dict(os.environ, **payload.get('env', {}))
    def execute(command):
        print('Executing: ' + ' '.join(command), flush=True)
        try:
            return subprocess.run(command, cwd=payload['cwd'], env=env,
                                  stdin=subprocess.DEVNULL).returncode
        except OSError as exc:
            print(str(exc), flush=True)
            return 127
    for step in payload['steps']:
        def update(phase):
            write_json(directory / 'progress.json', {'step': step['name'], 'phase': phase})
            print(f"{step['name']}: {phase}", flush=True)
        update('checking')
        if execute(step['check']) == 0:
            update('already ready')
            continue
        update('installing')
        if execute(step['setup']) != 0:
            update('setup failed')
            return 1
        update('verifying')
        if execute(step['check']) != 0:
            update('verification failed')
            return 1
        update('ready')
    return 0



def repository_root(value):
    root = Path(value).resolve()
    if not root.is_dir() or not (root / '.git').exists():
        raise ValueError(f'Remote repository unavailable (directory or .git missing): {root}')
    return root


def source_path(root, name):
    path = Path(name)
    if (not path.parts or path.is_absolute() or any(part.startswith(".") for part in path.parts)):
        raise ValueError(f"Invalid source path: {name}")
    target = root / path
    if target.resolve() != target:
        raise ValueError(f'Source path escapes repository or is a symlink: {name}')
    return target


def previous_manifest(root):
    marker = root / '.distribution-source.json'
    if marker.is_symlink():
        raise ValueError('Repository sync marker must not be a symlink')
    try:
        return json.loads(marker.read_text()).get('manifest', {})
    except FileNotFoundError:
        return {}


def repository_probe(payload):
    root = repository_root(payload['cwd'])
    changed = []
    for name, expected in payload['manifest'].items():
        path = source_path(root, name)
        if (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']
                or path.stat().st_mode & 0o777 != expected['mode']):
            changed.append(name)
    obsolete = sorted(set(previous_manifest(root)) - set(payload['manifest']))
    return dict(changed=changed, obsolete=obsolete)


def repository_update(payload):
    root = repository_root(payload['cwd'])
    old = previous_manifest(root)
    for name in payload.get('obsolete', []):
        if name not in old or name in payload['manifest']:
            raise ValueError(f'Refusing to delete unmanaged source: {name}')
        path = source_path(root, name)
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != old[name]["sha256"]:
                raise ValueError(f"Refusing to delete locally modified source: {name}")
            path.unlink()
    for name, value in payload['files'].items():
        expected = payload['manifest'][name]
        data = base64.b64decode(value['data'], validate=True)
        if hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        path = source_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.distribution-tmp')
        if temporary.is_symlink():
            raise ValueError(f'Unsafe temporary source path: {temporary}')
        with temporary.open('wb') as stream:
            stream.write(data)
        temporary.chmod(expected['mode'])
        temporary.replace(path)
    result = repository_probe(payload)
    if result['changed']:
        raise ValueError('Repository still differs after update: ' + ', '.join(result['changed']))
    marker = root / '.distribution-source.json'
    if marker.is_symlink():
        raise ValueError('Repository sync marker must not be a symlink')
    write_json(marker, dict(manifest=payload['manifest'], digest=payload['digest']))
    return dict(updated=len(payload['files']), removed=len(payload.get('obsolete', [])), digest=payload['digest'])


def main():
    action = sys.argv[1]
    payload = json.loads(sys.argv[2]) if action in ('supervise', 'provision') else json.load(sys.stdin)
    if action == 'preflight':
        try:
            os.chdir(payload['cwd'])
        except OSError as exc:
            print(json.dumps({'cwd_error': str(exc)}))
        else:
            print(json.dumps({'ok': True}))
    elif action == 'repo_probe':
        print(json.dumps(repository_probe(payload)))
    elif action == 'repo_update':
        print(json.dumps(repository_update(payload)))
    elif action == 'probe':
        print(json.dumps(machine_probe(payload['attempts'])))
    elif action == 'reap':
        print(json.dumps(reap(payload['attempt_dir'])))
    elif action == 'start':
        # Run this same helper independently of the SSH session.
        source = payload.pop('_source')
        directory = Path(payload['attempt_dir'])
        if 'steps' in payload:
            recipe = dict(payload)
            payload['command'] = [sys.executable, '-c', source, 'provision', json.dumps(recipe)]
        with (directory / 'supervisor.log').open('ab') as log:
            try:
                supervisor = subprocess.Popen([sys.executable, '-c', source, 'supervise', json.dumps(payload)],
                                              start_new_session=True, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=log)
                write_json(directory / 'supervisor.json', dict(
                    pid=supervisor.pid, start=process_start(supervisor.pid), boot_id=boot_id(),
                    host=os.uname().nodename))
            except Exception as exc:
                write_json(directory / 'status.json', {'state': 'failed', 'reason': str(exc)})
                raise
        print('{}')
    elif action == 'supervise':
        supervise(payload)
    elif action == 'provision':
        raise SystemExit(provision(payload))
    else:
        raise ValueError('Unknown remote action')


if __name__ == '__main__':
    main()
