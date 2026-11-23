"""Local controller-machine execution without loopback SSH authentication."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from device_jobs.controller import SSH, load_config


class LocalTransportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self.fixture.cleanup)
        self.config_path = Path(self.fixture.name) / 'config.json'
        self.config_path.write_text(json.dumps({
            'schema_version': 1,
            'shared_run_root': str(Path(self.fixture.name) / 'runs'), 'subjects': ['one'],
            'task': {'command': [sys.executable, '-c', 'pass', '{subject}', '{output_dir}']},
            'machines': [
                {'name': 'local-machine', 'host': 'localhost', 'transport': 'local',
                 'bootstrap_python': sys.executable, 'cwd': self.fixture.name,
                 'gpus': [{'id': 0, 'workers': 1}]},
                {'name': 'remote-worker', 'host': 'worker.example', 'user': 'worker',
                 'cwd': self.fixture.name, 'gpus': [{'id': 0, 'workers': 1}]},
            ],
        }))
        self.config = load_config(self.config_path)
        self.local = next(m for m in self.config['machines'] if m['name'] == 'local-machine')
        self.ssh = next(m for m in self.config['machines'] if m['name'] == 'remote-worker')

    def test_local_probe_invokes_python_directly_without_ssh(self):
        result = subprocess.CompletedProcess([], 0, '{}', '')
        with patch('device_jobs.controller.subprocess.run', return_value=result) as run:
            SSH(self.config).call(self.local, 'probe', {'attempts': []})
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[0], self.local['bootstrap_python'])
        self.assertEqual(arguments[1], '-c')
        self.assertEqual(arguments[-1], 'probe')
        self.assertNotIn('ssh', arguments)
        self.assertNotIn(self.local['host'], arguments)

    def test_remote_worker_still_uses_ssh(self):
        result = subprocess.CompletedProcess([], 0, '{}', '')
        with patch('device_jobs.controller.subprocess.run', return_value=result) as run:
            SSH(self.config).call(self.ssh, 'probe', {'attempts': []})
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[0], 'ssh')
        self.assertIn(self.ssh['user'] + '@' + self.ssh['host'], arguments)

    def test_local_launch_uses_same_real_detached_supervisor_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'heartbeat').touch()
            machine = dict(self.local, bootstrap_python=sys.executable)
            payload = dict(attempt_dir=tmp, lease_seconds=60, cwd=tmp, env={},
                           command=[sys.executable, '-c', 'print("local subject finished")'])
            SSH(self.config).call(machine, 'start', payload)
            deadline = time.monotonic() + 5
            status = {}
            while time.monotonic() < deadline:
                try:
                    status = json.loads((directory / 'status.json').read_text())
                except (OSError, ValueError):
                    pass
                if status.get('state') in ('succeeded', 'failed', 'cancelled'):
                    break
                time.sleep(.02)
            self.assertEqual(status['state'], 'succeeded')
            self.assertIn('local subject finished', (directory / 'job.log').read_text())

    def test_local_worker_does_not_inherit_shell_conda_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'heartbeat').touch()
            machine = dict(self.local, bootstrap_python=sys.executable)
            command = [sys.executable, '-c',
                       'import os; assert os.environ["CONDA_SHLVL"] == "0"; '
                       'assert not any(k == "CONDA_PREFIX" or k.startswith(("CONDA_PREFIX_", '
                       '"CONDA_STACKED_")) for k in os.environ); '
                       'assert os.environ["VCI_TEST_VALUE"] == "configured"; print("clean activation")']
            payload = dict(attempt_dir=tmp, lease_seconds=60, cwd=tmp,
                           env={'VCI_TEST_VALUE': 'configured'}, command=command)
            inherited = {'CONDA_SHLVL': '4', 'CONDA_PREFIX': '/unrelated/neupig',
                         'CONDA_PREFIX_1': '/unrelated/base', 'CONDA_PREFIX_3': '/unrelated/converter',
                         'CONDA_STACKED_4': 'true', 'CONDA_DEFAULT_ENV': 'neupig',
                         'CONDA_PROMPT_MODIFIER': '(neupig) '}
            with patch.dict(os.environ, inherited):
                SSH(self.config).call(machine, 'start', payload)
                self.assertEqual(os.environ['CONDA_SHLVL'], '4')
            deadline = time.monotonic() + 5
            status = {}
            while time.monotonic() < deadline:
                try:
                    status = json.loads((directory / 'status.json').read_text())
                except (OSError, ValueError):
                    pass
                if status.get('state') in ('succeeded', 'failed', 'cancelled'):
                    break
                time.sleep(.02)
            self.assertEqual(status.get('state'), 'succeeded', (directory / 'job.log').read_text())

    def test_invalid_transport_or_duplicate_local_machine_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            config = json.loads(self.config_path.read_text())
            config['machines'][1]['transport'] = 'local'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)
            config['machines'][1]['transport'] = 'unsupported'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
