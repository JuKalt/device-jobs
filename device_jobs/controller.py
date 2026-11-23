"""Distribute one command per subject across local or SSH-accessible Linux GPUs."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import threading
import uuid

from . import remote
from .dashboard import Dashboard, tail
from .repository import snapshot as repository_snapshot

TERMINAL = ('succeeded', 'failed', 'cancelled')
# EX_TEMPFAIL: the task could not run now (e.g. its subject lock is held by
# another process) and should be retried later rather than counted as failed.
RETRY_LATER_EXIT = 75


@dataclass
class Attempt:
    subject: str
    machine: dict
    gpu: dict
    directory: Path
    number: int
    cancel_requested: bool = False
    started: float = field(default_factory=time.monotonic)
    unreachable_since: float | None = None
    blocked_since: float | None = None


def positive(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be positive')
    if integer and not isinstance(value, int):
        raise ValueError(f'{label} must be an integer')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    subjects = config.get('subjects')
    if not isinstance(subjects, list) or not subjects or any(
        not isinstance(s, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', s) for s in subjects
    ) or len(set(subjects)) != len(subjects):
        raise ValueError('subjects must be unique safe subject IDs')
    task = config.get('task', {})
    command = task.get('command')
    if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
        raise ValueError('task.command must be an argument list')
    if not any('{subject}' in s or '{manifest}' in s for s in command):
        raise ValueError('task.command must select one subject using {subject} or {manifest}')
    if not any('{output_dir}' in s for s in command):
        raise ValueError('task.command must use {output_dir} for task outputs')
    if any(key in json.dumps(config).lower() for key in ('"password"', '"private_key"')):
        raise ValueError('Use SSH config/agent or identity_file; do not embed credentials in JSON')
    root = Path(config.get('shared_run_root', ''))
    if not root.is_absolute():
        raise ValueError('shared_run_root must be an absolute shared path')
    if 'output_root' in config:
        output = Path(config['output_root'])
        if not output.is_absolute() or output.resolve() == Path('/'):
            raise ValueError('output_root must be an absolute output directory other than /')
        output = output.resolve()
        logs = root.resolve()
        if output == logs or output in logs.parents or logs in output.parents:
            raise ValueError('output_root and shared_run_root must be separate directories')
    if task.get('output_layout', 'subject') not in ('subject', 'root'):
        raise ValueError('task.output_layout must be subject or root')
    if 'manifest' in task and not Path(task['manifest']).is_absolute():
        raise ValueError('task.manifest must be an absolute shared path')
    project = config.get("source")
    if project is not None:
        if not isinstance(project, dict) or set(project) != {"root", "include"}:
            raise ValueError("source must contain root and include")
        if not isinstance(project["root"], str) or not Path(project["root"]).is_absolute():
            raise ValueError("source.root must be an absolute checkout path")
        includes = project["include"]
        if not isinstance(includes, list) or not includes or any(
            not isinstance(item, str) or not item for item in includes
        ):
            raise ValueError("source.include must be a nonempty list of tracked-file patterns")
    if config.get("sync_repository"):
        raise ValueError("Replace legacy sync_repository with an explicit source.root and source.include")
    config.setdefault('poll_seconds', 5)
    config.setdefault('lease_seconds', 120)
    config.setdefault('ssh_timeout_seconds', 20)
    config.setdefault('busy_memory_fraction', .1)
    config.setdefault('busy_confirmation_polls', 2)
    config.setdefault('max_attempts', 5)
    config.setdefault('max_interruptions', 10)
    config.setdefault('provision_timeout_seconds', 7200)
    if task.get('timeout_seconds') is not None:
        positive(task['timeout_seconds'], 'task.timeout_seconds')
    positive(config['provision_timeout_seconds'], 'provision_timeout_seconds')
    positive(config['busy_confirmation_polls'], 'busy_confirmation_polls', integer=True)
    for key in ('poll_seconds', 'lease_seconds', 'ssh_timeout_seconds', 'max_attempts', 'max_interruptions'):
        positive(config[key], key, integer=key in ('max_attempts', 'max_interruptions'))
    if config['lease_seconds'] <= 2 * (config['ssh_timeout_seconds'] + config['poll_seconds']):
        raise ValueError('lease_seconds must exceed twice (ssh_timeout_seconds + poll_seconds)')
    if not isinstance(config['busy_memory_fraction'], (int, float)) or not 0 <= config['busy_memory_fraction'] < 1:
        raise ValueError('busy_memory_fraction must be between 0 and 1')
    machines = config.get('machines')
    if not isinstance(machines, list) or not machines:
        raise ValueError('machines must be a nonempty list')
    names = set()
    endpoints = set()
    local_count = 0
    for machine in machines:
        machine.setdefault('transport', 'ssh')
        if machine['transport'] not in ('local', 'ssh'):
            raise ValueError('machine.transport must be local or ssh')
        if machine['transport'] == 'local':
            local_count += 1
            if local_count > 1:
                raise ValueError('Only one machine can use local transport')
        for key in ('name', 'host', 'cwd'):
            if not isinstance(machine.get(key), str) or not machine[key]:
                raise ValueError(f'machine.{key} is required')
        if machine['host'].startswith('-') or any(c.isspace() for c in machine['host']):
            raise ValueError('Invalid SSH host')
        if not Path(machine['cwd']).is_absolute():
            raise ValueError('machine.cwd must be absolute')
        if machine['name'] in names or machine['host'] in endpoints:
            raise ValueError('Machine names and SSH hosts must be unique')
        names.add(machine['name'])
        endpoints.add(machine['host'])
        machine.setdefault('python', 'python3')
        machine.setdefault('bootstrap_python', 'python3')
        if not isinstance(machine['bootstrap_python'], str) or not machine['bootstrap_python']:
            raise ValueError('bootstrap_python must be an existing system Python executable')
        steps = machine.get('provision', config.get('provision', []))
        if not isinstance(steps, list):
            raise ValueError('provision must be a list of check/setup steps')
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get('name'), str):
                raise ValueError('Each provision step requires a name')
            for key in ('check', 'setup'):
                command = step.get(key)
                if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
                    raise ValueError(f'provision.{key} must be an argument array')
        provision_recipe(config, machine, Path('/validation'))
        if not isinstance(machine['python'], str) or not machine['python']:
            raise ValueError('machine.python must be an executable path')
        if 'user' in machine and (not isinstance(machine['user'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', machine['user'])):
            raise ValueError('Invalid SSH user')
        if 'port' in machine:
            positive(machine['port'], 'SSH port', integer=True)
            if machine['port'] > 65535:
                raise ValueError('Invalid SSH port')
        gpus = machine.get('gpus')
        if not isinstance(gpus, list) or not gpus:
            raise ValueError('Each machine requires GPU definitions')
        ids = set()
        for gpu in gpus:
            gpu['id'] = str(gpu['id'])
            if not re.fullmatch(r'[0-9]+|GPU-[A-Za-z0-9-]+', gpu['id']) or gpu['id'] in ids:
                raise ValueError('GPU IDs must be unique indices or full GPU UUIDs')
            ids.add(gpu['id'])
            gpu.setdefault('workers', 1)
            positive(gpu['workers'], 'gpu.workers', integer=True)
            for owner in (task, machine, gpu):
                env = owner.get('env', {})
                if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
                    raise ValueError('env must map strings to strings')
                params = owner.get('params', {})
                if not isinstance(params, dict) or any(not isinstance(k, str) or isinstance(v, (dict, list)) for k, v in params.items()):
                    raise ValueError('params must map names to scalar values')
    # Validate every template before any remote command can run.
    for machine in machines:
        for gpu in machine['gpus']:
            render(config, machine, gpu, subjects[0], root / 'validation')
    return config



def subject_output(config, subject, directory):
    if config.get('output_root'):
        return Path(config['output_root']) / subject
    return directory / 'output'


def command_output(config, subject, directory):
    if config.get('output_root') and config['task'].get('output_layout') == 'root':
        return Path(config['output_root'])
    return subject_output(config, subject, directory)


def render(config, machine, gpu, subject, directory):
    task = config['task']
    values = {}
    env = {}
    for owner in (task, machine, gpu):
        values.update(owner.get('params', {}))
        env.update(owner.get('env', {}))
    values.update(subject=subject, gpu_id=gpu['id'], device='0',
                  output_dir=str(command_output(config, subject, directory)), attempt_dir=str(directory),
                  subject_output_dir=str(subject_output(config, subject, directory)),
                  manifest=str(directory / 'manifest.json'), python=machine['python'], cwd=machine['cwd'])
    try:
        command = [s.format_map(values) for s in task['command']]
        env = {key: value.format_map(values) for key, value in env.items()}
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Invalid command/environment template: {exc}') from exc
    env.update(CUDA_VISIBLE_DEVICES=gpu.get('_uuid', gpu['id']), CUDA_DEVICE_ORDER='PCI_BUS_ID', VCI_SUBJECT_RUN_ID=subject,
               VCI_REMOTE_ATTEMPT_DIR=str(directory), PYTHONUNBUFFERED='1')
    return command, env


def provision_recipe(config, machine, directory):
    values = dict(config['task'].get('params', {}), **machine.get('params', {}))
    values.update(cwd=machine['cwd'], python=machine.get('python', 'python3'),
                  attempt_dir=str(directory))
    steps = []
    try:
        for step in machine.get('provision', config.get('provision', [])):
            steps.append(dict(name=step['name'],
                              check=[v.format_map(values) for v in step['check']],
                              setup=[v.format_map(values) for v in step['setup']]))
        env = dict(config['task'].get('env', {}), **machine.get('env', {})) if steps else {}
        env = {k: v.format_map(values) for k, v in env.items()}
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Invalid provisioning template: {exc}') from exc
    return dict(steps=steps, env=env, cwd=machine['cwd'], attempt_dir=str(directory),
                lease_seconds=config['lease_seconds'], timeout_seconds=config['provision_timeout_seconds'])


class SSH:
    def __init__(self, config):
        self.config = config
        self.source = Path(remote.__file__).read_text()

    def call(self, machine, action, payload):
        helper = [machine.get('bootstrap_python', 'python3'), '-c', self.source, action]
        local = machine.get('transport', 'ssh') == 'local'
        helper_env = None
        if local:
            arguments = helper
            # A screen shell may have several Conda environments stacked. The
            # helper is an independent worker, so conda run must not deactivate
            # that shell's environments (and execute their compiler hooks).
            helper_env = {key: value for key, value in os.environ.items()
                          if not (key == 'CONDA_PREFIX' or key.startswith('CONDA_PREFIX_')
                                  or key.startswith('CONDA_STACKED_')
                                  or key in ('CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER'))}
            helper_env['CONDA_SHLVL'] = '0'
        else:
            arguments = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                         '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
                         '-o', 'ServerAliveCountMax=2']
            if 'port' in machine:
                arguments += ['-p', str(machine['port'])]
            if 'identity_file' in machine:
                arguments += ['-i', str(Path(machine['identity_file']).expanduser())]
            target = machine['host']
            if 'user' in machine:
                target = machine['user'] + '@' + target
            arguments += [target, shlex.join(helper)]
        if action == 'start':
            payload = dict(payload, _source=self.source)
        result = subprocess.run(arguments, input=json.dumps(payload), capture_output=True,
                                text=True, timeout=self.config['ssh_timeout_seconds'], env=helper_env)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f'Control helper exited {result.returncode}')
        return json.loads(result.stdout)


def worker_preflight(config):
    """Check the Python helper and working directory once per worker."""
    workers = config['machines']
    transport = SSH(config)

    def probe(machine):
        try:
            result = transport.call(machine, 'preflight', {'cwd': machine['cwd']})
            if isinstance(result, dict) and isinstance(result.get('cwd_error'), str):
                return (f'Cannot enter the configured working directory ({result["cwd_error"]}). '
                        'Set machines[].cwd to an existing directory accessible on the worker.')
            if result != {'ok': True}:
                return 'Unexpected response from the remote Python helper. Check bootstrap_python.'
        except subprocess.TimeoutExpired:
            return 'Worker preflight timed out. Check the connection and ssh_timeout_seconds.'
        except json.JSONDecodeError:
            return 'Remote helper did not return JSON. Check bootstrap_python and SSH login output.'
        except (OSError, RuntimeError, ValueError) as exc:
            return str(exc)
        return None

    failed = False
    with ThreadPoolExecutor(max_workers=min(8, len(workers))) as pool:
        for machine, error in zip(workers, pool.map(probe, workers)):
            if error is None:
                continue
            failed = True
            print(f'{machine["name"]}: {machine["transport"].upper()} preflight failed: {error}', file=sys.stderr)
            details = error.lower()
            if 'host key verification failed' in details or 'remote host identification has changed' in details:
                print('  Verify the host fingerprint with your administrator before updating known_hosts.', file=sys.stderr)
            elif any(text in details for text in ('permission denied (', 'authentication failed',
                                                   'too many authentication failures')):
                identity = machine.get('identity_file')
                key = shlex.quote(str(Path(identity).expanduser())) if identity else '~/.ssh/id_ed25519'
                print('  In the terminal that runs Device Jobs, start an agent if needed:', file=sys.stderr)
                print('    eval "$(ssh-agent -s)"', file=sys.stderr)
                print(f'    ssh-add {key}', file=sys.stderr)
                print('  Use your existing SSH key and verify that the worker account accepts it.', file=sys.stderr)
    if failed:
        print('No jobs were started. Fix the worker errors and rerun, or use --preflight-only to recheck.', file=sys.stderr)
    return not failed


def read_status_file(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def read_status(directory):
    try:
        return json.loads((directory / 'status.json').read_text())
    except (OSError, ValueError):
        return {'state': 'unknown'}


def foreign_fraction(measurement):
    """None when the GPU or its occupancy could not be measured ([N/A] values)."""
    return None if not measurement else measurement.get('foreign_fraction')


def acquire_run_lock(output_root, run_dir):
    """One controller per output root: concurrent runs would contend for subjects
    and treat each other's GPU reservations as foreign work."""
    path = Path(output_root) / '.distribution.lock'
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        owner = stream.read().strip() or 'owner metadata unavailable'
        stream.close()
        raise ValueError(f'Another distribution controller is using output_root {output_root}: {owner}') from None
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps(dict(host=os.uname().nodename, pid=os.getpid(), run_dir=str(run_dir))))
    stream.flush()
    return stream


class Controller:
    def __init__(self, config, transport=None, dashboard=None):
        self.config = config
        self.transport = transport or SSH(config)
        # Unreachable or unresponsive workers have killed their jobs by now
        # (lease expiry); a missing supervisor after this long never started.
        self.lost_after = 2 * config['lease_seconds'] + config['poll_seconds']
        self.launch_grace = config['lease_seconds'] + 2 * config['ssh_timeout_seconds']
        self.pending = deque(config['subjects'])
        self.active = []
        self.attempt_counts = dict.fromkeys(config['subjects'], 0)
        self.failure_counts = dict.fromkeys(config['subjects'], 0)
        self.interruption_counts = dict.fromkeys(config['subjects'], 0)
        self.results = {}
        self.excluded = set()
        self.last_availability = {}
        self.measurements = {}
        self.busy_counts = {}
        self.connections = {}
        self.machine_states = {m['name']: ('pending setup' if m.get('provision', config.get('provision')) else 'ready')
                               for m in config['machines']}
        source = config.get("source")
        self.repository = repository_snapshot(source["root"], source["include"]) if source else None
        self.repositories_checked = set()
        self.setup_active = None
        self.setup_directories = {}
        self.root = Path(config['shared_run_root']) / ('run-' + uuid.uuid4().hex)
        self.run_lock = None
        if config.get('output_root'):
            Path(config['output_root']).mkdir(parents=True, exist_ok=True)
            self.run_lock = acquire_run_lock(config['output_root'], self.root)
        self.root.mkdir(parents=True)
        self.manifest = None
        if config['task'].get('manifest'):
            self.manifest = json.loads(Path(config['task']['manifest']).read_text())
            self.manifest_subjects = {str(s['subject_id']): s for s in self.manifest['subjects']}
            if any(s not in self.manifest_subjects for s in config['subjects']):
                raise ValueError('Configured subject missing from task.manifest')
        elif any('{manifest}' in s for s in config['task']['command']):
            raise ValueError('{manifest} requires task.manifest')
        remote.write_json(self.root / 'config.json', config)
        self.dashboard = Dashboard(self.root, self.snapshot, enabled=dashboard)
        remote.write_json(self.root / 'results.json', self.results)
        self.persist_queue()

    def report(self, message):
        self.dashboard.report(message)

    def snapshot(self):
        machines = []
        for machine in self.config['machines']:
            item = dict(name=machine['name'], host=machine['host'], transport=machine.get('transport', 'ssh'),
                        state=self.machine_states[machine['name']], gpus=[])
            if item['state'] == 'ready' and self.connections.get(machine['name']):
                item['state'] = 'unreachable: ' + self.connections[machine['name']]
            directory = self.setup_directories.get(machine['name'])
            if directory:
                progress = read_status_file(directory / 'progress.json')
                item['setup_detail'] = f"{progress.get('step', '')}: {progress.get('phase', '')} " + tail(directory / 'job.log')
            for gpu in machine['gpus']:
                key = (machine['name'], gpu['id'])
                jobs = []
                for attempt in list(self.active):
                    if attempt.machine is machine and attempt.gpu is gpu:
                        jobs.append(dict(subject=attempt.subject,
                                         state='stopping' if attempt.cancel_requested else read_status(attempt.directory)['state'],
                                         elapsed_seconds=int(time.monotonic() - attempt.started),
                                         detail=tail(attempt.directory / 'job.log')))
                state = self.last_availability.get(key, 'waiting for probe')
                if item['state'] != 'ready':
                    state = item['state']
                elif jobs and state == 'available':
                    state = 'running'
                item['gpus'].append(dict(id=gpu['id'], workers=gpu['workers'], state=state,
                                        memory=self.measurements.get(key, {}), jobs=jobs))
            machines.append(item)
        results = list(self.results.values())
        return dict(run_dir=str(self.root), output_root=self.config.get('output_root'), total=len(self.config['subjects']),
                    completed=sum(r['state'] == 'succeeded' for r in results),
                    failed=sum(r['state'] == 'failed' for r in results),
                    pending=len(self.pending), machines=machines)

    def synchronize_repository(self, machine):
        self.machine_states[machine['name']] = 'checking repository'
        self.report(f"{machine['name']}: comparing repository source")
        request = dict(cwd=machine['cwd'], manifest=self.repository['manifest'], digest=self.repository['digest'])
        changes = self.transport.call(machine, 'repo_probe', request)
        changed = changes.get('changed', [])
        obsolete = changes.get('obsolete', [])
        if changed or obsolete:
            self.machine_states[machine['name']] = 'updating repository'
            self.report(f"{machine['name']}: updating {len(changed)} source files, removing {len(obsolete)} obsolete managed files")
            result = self.transport.call(machine, 'repo_update', dict(request,
                files={name: self.repository['files'][name] for name in changed}, obsolete=obsolete))
            self.report(f"{machine['name']}: repository updated and verified ({result.get('updated', len(changed))} files)")
        else:
            self.report(f"{machine['name']}: repository source matches")
        self.repositories_checked.add(machine['name'])

    def check_repositories(self):
        for machine in self.config['machines']:
            if machine['name'] in self.repositories_checked:
                continue
            original = self.machine_states[machine['name']]
            try:
                self.synchronize_repository(machine)
                self.machine_states[machine['name']] = original
            except Exception as exc:
                self.repositories_checked.add(machine['name'])
                self.machine_states[machine['name']] = 'setup failed'
                self.report(f"{machine['name']}: repository sync failed: {exc}")

    def provision_tick(self):
        """Configure machines one at a time (they may share installation
        directories); ready machines keep scheduling meanwhile."""
        if self.setup_active:
            setup = self.setup_active
            machine, directory = setup.machine, setup.directory
            status = read_status(directory)
            if status['state'] in TERMINAL:
                state = 'ready' if status['state'] == 'succeeded' else 'setup failed'
                self.machine_states[machine['name']] = state
                reason = f" ({status['reason']})" if status.get('reason') else ''
                self.report(f"{machine['name']}: {state}{reason}; logs: {directory}")
                self.setup_active = None
            else:
                try:
                    result = self.transport.call(machine, 'probe', {'attempts': [str(directory)]})
                except Exception as exc:
                    self.machine_states[machine['name']] = 'setup connection lost: ' + str(exc)
                    self.check_liveness(setup, None, str(exc))
                    return
                devices = result.get('devices', {})
                for gpu in machine['gpus']:
                    if gpu['id'] in devices:
                        self.measurements[(machine['name'], gpu['id'])] = devices[gpu['id']]
                fraction = foreign_fraction(devices.get(setup.gpu['id']))
                if fraction is not None and fraction > self.config['busy_memory_fraction']:
                    (directory / 'cancel').touch()
                    self.machine_states[machine['name']] = 'stopping setup: GPU unavailable'
                elif not (directory / 'cancel').exists():
                    # Unknown occupancy ([N/A]) never stops a running setup.
                    (directory / 'heartbeat').touch()
                    self.machine_states[machine['name']] = 'configuring environment'
                self.check_liveness(setup, result.get('supervisors', {}).get(str(directory)), None)
                return
        machine = next((m for m in self.config['machines'] if self.machine_states[m['name']] == 'pending setup'), None)
        if machine is None:
            return
        try:
            devices = self.transport.call(machine, 'probe', {'attempts': []}).get('devices', {})
            for gpu in machine['gpus']:
                if gpu['id'] in devices:
                    self.measurements[(machine['name'], gpu['id'])] = devices[gpu['id']]
            selected = next((g for g in machine['gpus'] if foreign_fraction(devices.get(g['id'])) is not None
                             and foreign_fraction(devices[g['id']]) <= self.config['busy_memory_fraction']), None)
            if selected is None:
                raise RuntimeError('no idle GPU available for setup validation')
        except Exception as exc:
            self.machine_states[machine['name']] = 'setup failed'
            self.report(f"{machine['name']}: setup preflight failed: {exc}")
            return
        directory = self.root / 'environments' / f"machine-{self.config['machines'].index(machine)}"
        directory.mkdir(parents=True)
        self.setup_directories[machine['name']] = directory
        payload = provision_recipe(self.config, machine, directory)
        payload['env']['CUDA_VISIBLE_DEVICES'] = devices[selected['id']].get('uuid', selected['id'])
        payload['env']['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        payload['env']['PYTHONUNBUFFERED'] = '1'
        (directory / 'heartbeat').touch()
        remote.write_json(directory / 'status.json', {'state': 'starting'})
        remote.write_json(directory / 'recipe.json', payload)
        self.setup_active = Attempt(None, machine, selected, directory, 0)
        self.machine_states[machine['name']] = 'configuring environment'
        self.report(f"{machine['name']}: checking/configuring environments; logs: {directory}")
        try:
            self.transport.call(machine, 'start', payload)
        except Exception as exc:
            (directory / 'cancel').touch()
            self.machine_states[machine['name']] = 'setup launch uncertain'
            self.report(f"{machine['name']}: setup launch uncertain: {exc}; awaiting termination")

    def check_liveness(self, attempt, supervisor, error):
        """Finish attempts whose remote supervisor can no longer report a result.

        supervisor is the worker's view (alive, dead, missing) or None when the
        machine could not be reached (error)."""
        if read_status(attempt.directory)['state'] in TERMINAL:
            return
        now = time.monotonic()
        name = attempt.machine['name']
        if error is not None:
            attempt.unreachable_since = attempt.unreachable_since or now
            if now - attempt.unreachable_since > self.lost_after:
                # Leases are not renewed for unreachable machines, so any
                # surviving supervisor has stopped the job by now.
                remote.write_json(attempt.directory / 'status.json', {
                    'state': 'cancelled', 'reason': f'{name} unreachable for {int(now - attempt.unreachable_since)} s'})
            return
        attempt.unreachable_since = None
        if supervisor == 'alive' or supervisor is None:
            return
        if supervisor == 'missing' and now - attempt.started < self.launch_grace:
            return
        try:
            state = self.transport.call(attempt.machine, 'reap', {'attempt_dir': str(attempt.directory)}).get('state')
        except Exception as exc:
            self.report(f'{name}: could not stop leftovers of {attempt.directory}: {exc}')
            return
        if state != 'blocked':
            attempt.blocked_since = None
            if state == 'cancelled':
                self.report(f'{name}: supervisor of {attempt.directory} is gone; stopped its leftovers')
            return
        attempt.blocked_since = attempt.blocked_since or now
        if now - attempt.blocked_since > self.config['lease_seconds']:
            # Unkillable workers may still hold the GPU: stop using it.
            if attempt.subject is not None:
                self.excluded.add((name, attempt.gpu['id']))
            remote.write_json(attempt.directory / 'status.json', {
                'state': 'failed', 'reason': f'workers on {name} GPU {attempt.gpu["id"]} cannot be stopped; GPU excluded'})

    def cancel(self, attempt):
        (attempt.directory / 'cancel').touch()
        attempt.cancel_requested = True

    def collect(self):
        for attempt in list(self.active):
            status = read_status(attempt.directory)
            state = status['state']
            if state not in TERMINAL:
                continue
            self.active.remove(attempt)
            subject = attempt.subject
            succeeded = state == 'succeeded' and not attempt.cancel_requested
            if not succeeded and self.config.get('output_root'):
                try:
                    self.validate_subject_output(attempt)
                except ValueError as exc:
                    self.report(f'Failed {subject}: {exc}')
                    self.results[subject] = {'state': 'failed', 'attempt_dir': str(attempt.directory), 'reason': str(exc)}
                    remote.write_json(self.root / 'results.json', self.results)
                    continue
            interrupted = not succeeded and (attempt.cancel_requested or status.get('returncode') == RETRY_LATER_EXIT or (
                state == 'cancelled' and status.get('reason') != 'command timeout'))
            if interrupted:
                self.interruption_counts[subject] += 1
            elif not succeeded:
                self.failure_counts[subject] += 1
            reason = status.get('reason') or (
                'subject locked by another process' if status.get('returncode') == RETRY_LATER_EXIT else state)
            if succeeded:
                self.report(f'Completed {subject} on {attempt.machine["name"]} GPU {attempt.gpu["id"]}: {attempt.directory}')
                self.results[subject] = {'state': state, 'attempt_dir': str(attempt.directory),
                                         'output_dir': str(subject_output(self.config, subject, attempt.directory))}
            elif (self.interruption_counts[subject] <= self.config['max_interruptions'] if interrupted
                  else self.failure_counts[subject] < self.config['max_attempts']):
                self.pending.append(subject)
                self.report(f'Requeue {subject}: {reason}')
            else:
                limit = 'interruption limit reached' if interrupted else 'failure limit reached'
                self.report(f'Failed {subject}: {limit} ({reason}); logs: {attempt.directory}')
                self.results[subject] = {'state': 'failed', 'attempt_dir': str(attempt.directory),
                                         'reason': f'{limit}: {reason}'}
            remote.write_json(self.root / 'results.json', self.results)

    def tick(self):
        if self.repository and len(self.repositories_checked) < len(self.config['machines']):
            self.check_repositories()
        if self.setup_active or any(state == 'pending setup' for state in self.machine_states.values()):
            self.provision_tick()
        self.collect()
        machines = [m for m in self.config['machines'] if self.machine_states[m['name']] == 'ready'
                    or any(a.machine is m for a in self.active)]
        def check(machine):
            attempts = [str(a.directory) for a in self.active if a.machine is machine]
            try:
                return machine, self.transport.call(machine, 'probe', {'attempts': attempts}), None
            except Exception as exc:
                return machine, {}, str(exc)
        snapshots = []
        if machines:
            with ThreadPoolExecutor(max_workers=len(machines)) as pool:
                snapshots = list(pool.map(check, machines))
        launches = []
        for machine, result, error in snapshots:
            self.connections[machine['name']] = error
            devices = result.get('devices', {})
            for attempt in [a for a in self.active if a.machine is machine]:
                self.check_liveness(attempt, result.get('supervisors', {}).get(str(attempt.directory)), error)
            for gpu in machine['gpus']:
                key = (machine['name'], gpu['id'])
                measurement = devices.get(gpu['id'])
                if measurement:
                    self.measurements[key] = measurement
                if measurement and measurement.get('uuid'):
                    gpu['_uuid'] = measurement['uuid']
                fraction = foreign_fraction(measurement)
                busy = fraction is not None and fraction > self.config['busy_memory_fraction']
                self.busy_counts[key] = self.busy_counts.get(key, 0) + 1 if busy and not error else 0
                available = (not error and fraction is not None and not busy and key not in self.excluded
                             and self.machine_states[machine['name']] == 'ready')
                description = 'available' if available else (
                    error or ('busy' if busy else 'excluded' if key in self.excluded
                              else result.get('gpu_error') or ('GPU memory unreadable' if measurement else 'missing GPU')))
                if self.last_availability.get(key) != description:
                    self.report(f'{key[0]} GPU {key[1]}: {description}')
                    self.last_availability[key] = description
                attempts = [a for a in self.active if a.machine is machine and a.gpu is gpu]
                if busy and self.busy_counts[key] >= self.config['busy_confirmation_polls']:
                    # Busy GPUs remain eligible when external work releases them.
                    for attempt in attempts:
                        if not attempt.cancel_requested:
                            self.report(f'Cancel {attempt.subject} on {key[0]} GPU {key[1]}: '
                                        f'confirmed external memory {fraction:.1%} '
                                        f'over {self.busy_counts[key]} polls')
                            self.cancel(attempt)
                # A single busy sample prevents new launches, but existing jobs
                # keep their leases until external usage has been confirmed.
                if not error and measurement is not None:
                    for attempt in attempts:
                        if not attempt.cancel_requested:
                            (attempt.directory / 'heartbeat').touch()
                if available:
                    # Wait for cancelled workers to confirm termination before reuse.
                    if any(attempt.cancel_requested for attempt in attempts):
                        continue
                    for _ in range(gpu['workers'] - len(attempts)):
                        if not self.pending:
                            break
                        subject = self.pending.popleft()
                        try:
                            launches.append(self.prepare_launch(subject, machine, gpu))
                        except ValueError as exc:
                            self.report(f'Failed {subject}: {exc}')
                            self.results[subject] = {'state': 'failed', 'reason': str(exc)}
                            remote.write_json(self.root / 'results.json', self.results)
        if launches:
            # SSH handshakes run concurrently so worker counts do not delay heartbeats.
            with ThreadPoolExecutor(max_workers=len(launches)) as pool:
                outcomes = list(pool.map(self.dispatch, launches))
            for attempt, error in outcomes:
                if error:
                    self.excluded.add((attempt.machine['name'], attempt.gpu['id']))
                    self.cancel(attempt)
                    self.report(f'Launch uncertain for {attempt.subject}: {error}; awaiting remote termination confirmation')
        self.persist_queue()

    def persist_queue(self):
        remote.write_json(self.root / 'queue.json', {
            'pending': list(self.pending), 'active': [str(a.directory) for a in self.active],
            'machines': dict(self.machine_states),
            'devices': [{'machine': k[0], 'gpu': k[1], 'status': v} for k, v in self.last_availability.items()]})

    def validate_subject_output(self, attempt):
        # Preserve completed stages and checkpoints after confirmed termination.
        root = Path(self.config['output_root']).resolve()
        output = root / attempt.subject
        if output.is_symlink() or output.resolve().parent != root:
            raise ValueError(f'Refusing to resume a subject output symlink/outside directory: {output}')
        if output.exists():
            self.report(f'Preserved output for resume: {attempt.subject}: {output}')

    def prepare_launch(self, subject, machine, gpu):
        self.attempt_counts[subject] += 1
        number = self.attempt_counts[subject]
        directory = self.root / subject / f'attempt-{number}'
        directory.mkdir(parents=True)
        output = subject_output(self.config, subject, directory)
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError(f'Subject output must be a directory, not a file or symlink: {output}')
        if not self.config.get('output_root') or self.config['task'].get('output_layout', 'subject') == 'subject':
            output.mkdir(parents=True, exist_ok=True)
        (directory / 'heartbeat').touch()
        if self.manifest is not None:
            shard = dict(self.manifest, subjects=[self.manifest_subjects[subject]])
            remote.write_json(directory / 'manifest.json', shard)
        command, env = render(self.config, machine, gpu, subject, directory)
        payload = dict(command=command, env=env, cwd=machine['cwd'], attempt_dir=str(directory),
                       lease_seconds=self.config['lease_seconds'], timeout_seconds=self.config['task'].get('timeout_seconds'))
        remote.write_json(directory / 'job.json', dict(payload, machine=machine['name'], gpu=gpu['id']))
        remote.write_json(directory / 'status.json', {'state': 'starting'})
        attempt = Attempt(subject, machine, gpu, directory, number)
        self.active.append(attempt)
        self.report(f'Start {subject} attempt {number} on {machine["name"]} GPU {gpu["id"]}')
        return attempt, payload

    def dispatch(self, item):
        attempt, payload = item
        try:
            self.transport.call(attempt.machine, 'start', payload)
            return attempt, None
        except Exception as exc:
            # A lost acknowledgement never proves that the job did not start.
            return attempt, str(exc)

    def renew_leases(self):
        for attempt in list(self.active):
            if not attempt.cancel_requested and not self.connections.get(attempt.machine["name"]):
                (attempt.directory / 'heartbeat').touch()

    def keep_leases_alive(self, stop):
        # Lease renewal must continue during filesystem work and SSH calls.
        interval = min(5, self.config['lease_seconds'] / 4)
        while not stop.wait(interval):
            self.renew_leases()

    def run(self):
        self.report(f'Run state and logs: {self.root}')
        self.dashboard.start()
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(target=self.keep_leases_alive, args=(heartbeat_stop,), daemon=True)
        heartbeat_thread.start()
        try:
            while self.pending or self.active:
                self.tick()
                if self.pending and not self.active and all(
                    ((m['name'], g['id']) in self.excluded or self.machine_states[m['name']] == 'setup failed')
                    for m in self.config['machines'] for g in m['gpus']
                ):
                    self.report(f'All GPUs excluded; unfinished subjects remain in {self.root / "queue.json"}')
                    return 1
                if self.pending or self.active:
                    time.sleep(self.config['poll_seconds'])
        except KeyboardInterrupt:
            for attempt in self.active:
                self.cancel(attempt)
            self.report(f'Stop requested. Remote watchdogs will terminate active jobs; inspect {self.root}')
            return 130
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join()
            for attempt in self.active:
                self.cancel(attempt)
            if self.setup_active:
                (self.setup_active.directory / 'cancel').touch()
            self.dashboard.close()
            if self.run_lock:
                self.run_lock.close()
            self.report(f'Final state and logs: {self.root}')
        return 0 if all(r['state'] == 'succeeded' for r in self.results.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Validate and show commands without SSH or job execution')
    mode.add_argument('--preflight-only', action='store_true', help='Check worker access, Python and working directories without launching jobs')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--dashboard', dest='dashboard', action='store_true', help='Force the live terminal dashboard')
    display.add_argument('--no-dashboard', dest='dashboard', action='store_false', help='Plain terminal event logging')
    parser.set_defaults(dashboard=None)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            if config.get("source"):
                source = config["source"]
                bundle = repository_snapshot(source["root"], source["include"])
                print(f'Source snapshot: {len(bundle["manifest"])} files ({bundle["digest"]})')
            for machine in config['machines']:
                for gpu in machine['gpus']:
                    command, _ = render(config, machine, gpu, config['subjects'][0], Path(config['shared_run_root']) / 'preview')
                    print(f'{machine["name"]} ({machine.get("transport", "ssh")}) GPU {gpu["id"]}, {gpu["workers"]} concurrent subjects: {shlex.join(command)}')
                for step in provision_recipe(config, machine, Path(config['shared_run_root']) / 'preview')['steps']:
                    print(f'Setup {machine["name"]}: {step["name"]}; check: {shlex.join(step["check"])}; install: {shlex.join(step["setup"])}')
            return 0
        if not worker_preflight(config):
            return 1
        if args.preflight_only:
            print('Worker preflight passed.')
            return 0
        return Controller(config, dashboard=args.dashboard).run()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
