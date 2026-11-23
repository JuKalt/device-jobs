"""Live terminal status and durable controller events without extra dependencies."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import sys
import threading

from .remote import write_json


_ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')


def clean(value):
    return ''.join(c for c in _ANSI.sub('', str(value)) if c.isprintable())


def tail(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        return next((clean(line).strip() for line in reversed(lines) if clean(line).strip()), '')
    except OSError:
        return ''


def lines(snapshot, width=140, height=1000):
    result = [f"Subject distribution | completed {snapshot['completed']}/{snapshot['total']} "
              f"| failed {snapshot['failed']} | queued {snapshot['pending']}",
              f"Logs: {snapshot['run_dir']}"]
    if snapshot.get('output_root'):
        result.append(f"Outputs: {snapshot['output_root']}")
    base_rows = 3 + bool(snapshot.get('output_root')) + len(snapshot.get('events', [])[-4:]) + sum(
        1 + bool(m.get('setup_detail')) + len(m['gpus']) for m in snapshot['machines'])
    job_rows = sum(len(g['jobs']) for m in snapshot['machines'] for g in m['gpus'])
    details_fit = base_rows + job_rows < height
    for machine in snapshot['machines']:
        result.append(f"{machine['name']} ({machine['host']}, {machine.get('transport', 'ssh')}) | {machine['state']}")
        if machine.get('setup_detail'):
            result.append('  Setup: ' + machine['setup_detail'])
        for gpu in machine['gpus']:
            memory = gpu.get('memory', {})
            usage = ''
            def show(value, form):
                return 'n/a' if value is None else format(value, form)
            if 'total_mib' in memory:
                used = memory.get('used_mib')
                total = memory.get('total_mib')
                usage = (f" | VRAM {show(used and used / 1024, '.1f')}/{show(total and total / 1024, '.1f')} GiB"
                         f" ext {show(memory.get('foreign_fraction'), '.0%')}")
                if 'utilization' in memory:
                    usage += f" util {show(memory['utilization'], '.0f')}%"
            workers = gpu['jobs']
            summaries = []
            for job in workers:
                progress = re.search(r'\d+(?:\.\d+)?%', job['detail'])
                detail = progress[0] if progress else job['detail'][:16]
                summaries.append(f"{job['subject']}:{job['state']}" + (f"({detail})" if detail else ''))
            result.append(f"  GPU {gpu['id']}: {gpu['state']} | {len(workers)}/{gpu['workers']} slots{usage}"
                          + (' | ' + ', '.join(summaries) if workers else ' | waiting'))
            if details_fit:
                for job in workers:
                    result.append(f"    {job['subject']} [{job['state']}, {job['elapsed_seconds']}s] {job['detail']}")
    result.append('Recent events:')
    result.extend('  ' + event for event in snapshot.get('events', [])[-4:])
    return [clean(line)[:width] for line in result]


class Dashboard:
    def __init__(self, root, snapshot, enabled=None, stream=None):
        self.root = root
        self.snapshot = snapshot
        self.stream = stream or sys.stdout
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.events = deque(maxlen=50)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def report(self, message):
        event = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {clean(message)}"
        with self.lock:
            self.events.append(event)
            with (self.root / 'controller.log').open('a') as stream:
                stream.write(event + '\n')
            if not self.enabled or self.thread is None:
                print(message, file=self.stream, flush=True)

    def refresh(self):
        snapshot = self.snapshot()
        with self.lock:
            snapshot['events'] = list(self.events)
            write_json(self.root / 'dashboard.json', snapshot)
            if self.enabled:
                size = shutil.get_terminal_size((140, 40))
                width = max(1, size.columns - 1)
                self.stream.write('\x1b[H\x1b[2J' + '\n'.join(lines(snapshot, width, size.lines)) + '\n')
                self.stream.flush()

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except (OSError, ValueError, RuntimeError):
                # Temporary shared-drive failure must not hide watchdog handling.
                pass
            self.stop_event.wait(1)

    def start(self):
        if self.enabled:
            self.stream.write('\x1b[?1049h\x1b[?25l')
            self.stream.flush()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            self.refresh()
        finally:
            self.thread = None
            if self.enabled:
                self.stream.write('\x1b[?25h\x1b[?1049l')
                self.stream.flush()
