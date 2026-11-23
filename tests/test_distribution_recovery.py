"""Recovery of lost supervisors, retry caps, run locking and unreadable GPU data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from device_jobs import remote
from device_jobs.controller import RETRY_LATER_EXIT, Controller, load_config
from tests.test_distribution import FakeSSH


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            'schema_version': 1, 'sync_repository': False, 'shared_run_root': str(self.root / 'runs'),
            'output_root': str(self.root / 'processed'), 'subjects': ['one'],
            'task': {'command': ['python3', 'task.py', '{subject}', '{output_dir}']},
            'machines': [{'name': 'host', 'host': 'host', 'cwd': str(self.root), 'gpus': [{'id': 0, 'workers': 1}]}]}
        self.transport = FakeSSH()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def controller(self, **overrides):
        path = self.root / 'config.json'
        path.write_text(json.dumps(dict(self.config, **overrides)))
        controller = Controller(load_config(path), self.transport)
        self.addCleanup(lambda: controller.run_lock and controller.run_lock.close())
        return controller

    def test_dead_supervisor_is_reaped_and_subject_requeued(self):
        controller = self.controller()
        controller.tick()
        attempt = controller.active[0]
        self.transport.supervisors[str(attempt.directory)] = 'dead'
        controller.tick()
        self.assertEqual(self.transport.reaped, [str(attempt.directory)])
        controller.tick()
        self.assertEqual(controller.active[0].number, 2)
        self.assertEqual(controller.failure_counts['one'], 0)
        self.assertEqual(controller.interruption_counts['one'], 1)

    def test_missing_supervisor_waits_for_launch_grace(self):
        controller = self.controller()
        controller.tick()
        attempt = controller.active[0]
        remote.write_json(attempt.directory / 'status.json', {'state': 'starting'})
        self.transport.supervisors[str(attempt.directory)] = 'missing'
        controller.tick()
        self.assertEqual(self.transport.reaped, [])
        attempt.started -= controller.launch_grace + 1
        controller.tick()
        self.assertEqual(self.transport.reaped, [str(attempt.directory)])

    def test_unreachable_machine_releases_attempt_after_leases_expire(self):
        controller = self.controller()
        controller.tick()
        attempt = controller.active[0]
        self.transport.unreachable.add('host')
        controller.tick()
        self.assertIs(controller.active[0], attempt)
        attempt.unreachable_since -= controller.lost_after + 1
        controller.tick()
        status = json.loads((attempt.directory / 'status.json').read_text())
        self.assertEqual(status['state'], 'cancelled')
        self.assertIn('unreachable', status['reason'])
        controller.tick()
        self.assertEqual(controller.active, [])
        self.assertEqual(list(controller.pending), ['one'])

    def test_unstoppable_workers_fail_attempt_and_exclude_gpu(self):
        controller = self.controller()
        controller.tick()
        attempt = controller.active[0]
        self.transport.supervisors[str(attempt.directory)] = 'dead'
        self.transport.reap_state = 'blocked'
        controller.tick()
        self.assertIs(controller.active[0], attempt)
        attempt.blocked_since -= controller.config['lease_seconds'] + 1
        controller.tick()
        controller.tick()
        self.assertIn(('host', '0'), controller.excluded)
        self.assertEqual(controller.failure_counts['one'], 1)
        self.assertEqual(controller.active, [])

    def test_interruptions_are_capped(self):
        controller = self.controller(max_interruptions=2)
        controller.tick()
        for _ in range(3):
            remote.write_json(controller.active[0].directory / 'status.json',
                              {'state': 'cancelled', 'reason': 'controller lease expired'})
            controller.tick()
        self.assertEqual(controller.results['one']['state'], 'failed')
        self.assertIn('interruption limit', controller.results['one']['reason'])
        self.assertEqual(controller.failure_counts['one'], 0)

    def test_locked_subject_exit_is_an_interruption(self):
        controller = self.controller(max_attempts=1)
        controller.tick()
        remote.write_json(controller.active[0].directory / 'status.json',
                          {'state': 'failed', 'returncode': RETRY_LATER_EXIT})
        controller.tick()
        self.assertNotIn('one', controller.results)
        self.assertEqual(controller.failure_counts['one'], 0)
        self.assertEqual(controller.active[0].number, 2)

    def test_unknown_gpu_memory_blocks_launch_but_keeps_jobs(self):
        self.config['machines'][0]['gpus'][0]['workers'] = 2
        controller = self.controller()
        controller.tick()
        self.assertEqual(len(controller.active), 1)
        controller.pending.append('two')
        controller.attempt_counts['two'] = controller.failure_counts['two'] = controller.interruption_counts['two'] = 0
        heartbeat = controller.active[0].directory / 'heartbeat'
        os.utime(heartbeat, (100, 100))
        self.transport.fractions[('host', '0')] = None
        for _ in range(3):
            controller.tick()
        self.assertEqual(len(controller.active), 1)
        self.assertFalse(controller.active[0].cancel_requested)
        self.assertGreater(heartbeat.stat().st_mtime, 100)
        self.assertEqual(controller.last_availability[('host', '0')], 'GPU memory unreadable')

    def test_second_controller_on_same_output_root_is_refused(self):
        first = self.controller()
        script = ('import json, sys; from device_jobs.controller import Controller, load_config; '
                  'Controller(load_config(sys.argv[1]), object())')
        result = subprocess.run([sys.executable, '-c', script, str(self.root / 'config.json')],
                                capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Another distribution controller', result.stderr)
        self.assertIn(str(first.root), result.stderr)

    def test_task_timeout_reaches_supervisor(self):
        controller = self.controller(task=dict(self.config['task'], timeout_seconds=3600))
        controller.tick()
        self.assertEqual(self.transport.starts[0][1]['timeout_seconds'], 3600)

    def test_ready_machine_schedules_while_another_is_provisioned(self):
        step = {'name': 'runtime', 'check': ['true'], 'setup': ['true']}
        self.config['machines'].append({'name': 'new', 'host': 'new', 'cwd': str(self.root),
                                        'gpus': [{'id': 0}], 'provision': [step]})
        controller = self.controller()
        controller.tick()
        self.assertIsNotNone(controller.setup_active)
        self.assertEqual([a.machine['name'] for a in controller.active], ['host'])

    def test_dead_setup_supervisor_fails_setup(self):
        self.config['provision'] = [{'name': 'runtime', 'check': ['true'], 'setup': ['true']}]
        controller = self.controller()
        controller.tick()
        setup = controller.setup_active
        self.transport.supervisors[str(setup.directory)] = 'dead'
        controller.tick()
        controller.tick()
        self.assertEqual(controller.machine_states['host'], 'setup failed')
        self.assertIsNone(controller.setup_active)


class RemoteHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_unavailable_nvidia_smi_values_are_unknown_not_fatal(self):
        rows = [[['0', 'GPU-one', '[N/A]', '[N/A]', '[N/A]'], ['1', 'GPU-two', '1000', '100', '5']],
                [['GPU-two', '10', '[N/A]']],
                [['0', 'GPU-one', '[N/A]', '[N/A]', '[N/A]'], ['1', 'GPU-two', '1000', '100', '5']]]
        remote.write_json(self.directory / 'status.json', {'state': 'running', 'pgid': 1})
        with patch.object(remote, 'query', side_effect=rows), patch.object(os, 'getpgid', return_value=1):
            result = remote.probe([str(self.directory)])
        self.assertIsNone(result['0']['foreign_fraction'])
        self.assertIsNone(result['0']['utilization'])
        # Our own process's usage is unreadable: GPU-two must not look foreign.
        self.assertIsNone(result['1']['foreign_fraction'])

    def test_failed_gpu_query_still_reports_supervisors(self):
        with patch.object(remote, 'query', side_effect=subprocess.CalledProcessError(9, 'nvidia-smi')):
            result = remote.machine_probe([str(self.directory)])
        self.assertEqual(result['devices'], {})
        self.assertIn('nvidia-smi', result['gpu_error'])
        self.assertEqual(result['supervisors'], {str(self.directory): 'missing'})

    def test_supervisor_state_detects_live_dead_and_rebooted(self):
        record = dict(pid=os.getpid(), start=remote.process_start(os.getpid()), boot_id=remote.boot_id())
        remote.write_json(self.directory / 'supervisor.json', record)
        self.assertEqual(remote.supervisor_state(self.directory), 'alive')
        remote.write_json(self.directory / 'supervisor.json', dict(record, start=record['start'] + 1))
        self.assertEqual(remote.supervisor_state(self.directory), 'dead')
        remote.write_json(self.directory / 'supervisor.json', dict(record, boot_id='previous-boot'))
        self.assertEqual(remote.supervisor_state(self.directory), 'dead')

    def test_reap_finalizes_lost_attempt_and_keeps_terminal_results(self):
        remote.write_json(self.directory / 'status.json', {'state': 'starting'})
        self.assertEqual(remote.reap(self.directory), {'state': 'cancelled'})
        status = json.loads((self.directory / 'status.json').read_text())
        self.assertEqual(status['reason'], 'supervisor never started')
        remote.write_json(self.directory / 'status.json', {'state': 'succeeded', 'returncode': 0})
        self.assertEqual(remote.reap(self.directory), {'state': 'succeeded'})

    def test_reap_stops_orphaned_job_processes(self):
        env = dict(os.environ, VCI_REMOTE_ATTEMPT_DIR=str(self.directory))
        orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], env=env,
                                  start_new_session=True)
        self.addCleanup(lambda: orphan.poll() is None and orphan.kill())
        remote.write_json(self.directory / 'supervisor.json', dict(pid=orphan.pid, start=-1, boot_id=remote.boot_id()))
        remote.write_json(self.directory / 'status.json', {'state': 'running', 'pgid': orphan.pid})
        self.assertEqual(remote.reap(self.directory), {'state': 'cancelled'})
        self.assertIsNotNone(orphan.wait(timeout=10))
        self.assertEqual(json.loads((self.directory / 'status.json').read_text())['reason'], 'supervisor lost')

    def test_start_records_supervisor_identity(self):
        source = Path(remote.__file__).read_text()
        (self.directory / 'heartbeat').touch()
        payload = dict(attempt_dir=str(self.directory), lease_seconds=60, cwd=str(self.directory), env={},
                       command=[sys.executable, '-c', 'pass'], _source=source)
        subprocess.run([sys.executable, '-c', source, 'start'], input=json.dumps(payload),
                       text=True, check=True, capture_output=True)
        record = json.loads((self.directory / 'supervisor.json').read_text())
        self.assertEqual(record['boot_id'], remote.boot_id())
        self.assertIsInstance(record['start'], int)


if __name__ == '__main__':
    unittest.main()
