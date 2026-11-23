from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from device_jobs.controller import Controller, load_config, main
from device_jobs import remote


class FakeSSH:
    def __init__(self):
        self.busy = set()
        self.starts = []
        self.unreachable = set()
        self.uncertain = False
        self.probes = []
        self.fractions = {}
        self.supervisors = {}
        self.reaped = []
        self.reap_state = 'cancelled'

    def call(self, machine, action, payload):
        if machine['name'] in self.unreachable:
            raise TimeoutError('offline')
        if action == 'probe':
            self.probes.append(payload['attempts'])
            return dict(devices={g['id']: {'foreign_fraction': self.fractions.get(
                                     (machine['name'], g['id']), .2 if (machine['name'], g['id']) in self.busy else 0.)}
                                 for g in machine['gpus']},
                        supervisors={d: self.supervisors.get(d, 'alive') for d in payload['attempts']})
        if action == 'reap':
            self.reaped.append(payload['attempt_dir'])
            remote.write_json(Path(payload['attempt_dir']) / 'status.json', {'state': self.reap_state, 'reason': 'supervisor lost'})
            return {'state': self.reap_state}
        self.starts.append((machine, payload))
        if self.uncertain:
            raise TimeoutError('acknowledgement lost')
        remote.write_json(Path(payload['attempt_dir']) / 'status.json', {'state': 'running', 'pgid': 123})
        return {}


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = {
            'schema_version': 1, 'sync_repository': False, 'shared_run_root': str(self.root / 'runs'),
            'subjects': ['one', 'two', 'three'],
            'task': {'command': ['python3', 'task.py', '{subject}', '{output_dir}', '{batch}', '{device}'],
                     'params': {'batch': 8}},
            'machines': [{'name': 'host', 'host': 'host', 'cwd': str(self.root),
                          'gpus': [{'id': 0, 'workers': 2, 'params': {'batch': 16}},
                                   {'id': 1, 'workers': 1}]}]}
        self.path = self.root / 'config.json'
        self.path.write_text(json.dumps(config))
        self.config = load_config(self.path)
        self.transport = FakeSSH()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_failed_ssh_preflight_reports_all_workers_before_creating_run(self):
        config = json.loads(self.path.read_text())
        config['machines'] += [
            dict(config['machines'][0], name='key-rejected', host='key-rejected', identity_file='/keys/work key'),
            dict(config['machines'][0], name='local', host='localhost', transport='local'),
        ]
        self.path.write_text(json.dumps(config))

        def ssh_result(machine, action, payload):
            self.assertEqual(action, 'preflight')
            if machine['transport'] == 'local':
                return {'ok': True}
            errors = {'host': 'Host key verification failed.',
                      'key-rejected': 'Permission denied (publickey,password).'}
            raise RuntimeError(errors[machine['name']])

        errors = io.StringIO()
        with patch('device_jobs.controller.SSH.call', side_effect=ssh_result) as call:
            with contextlib.redirect_stderr(errors):
                self.assertEqual(main(['--config', str(self.path), '--dry-run']), 0)
                call.assert_not_called()
                self.assertEqual(main(['--config', str(self.path)]), 1)
        self.assertEqual(call.call_count, 3)
        self.assertIn('host: SSH preflight failed', errors.getvalue())
        self.assertIn('Verify the host fingerprint', errors.getvalue())
        self.assertIn('key-rejected: SSH preflight failed', errors.getvalue())
        self.assertIn("ssh-add '/keys/work key'", errors.getvalue())
        self.assertFalse((self.root / 'runs').exists())

    def test_successful_preflight_only_does_not_create_a_run(self):
        with patch('device_jobs.controller.SSH.call', return_value={'ok': True}) as call:
            self.assertEqual(main(['--config', str(self.path), '--preflight-only']), 0)
        call.assert_called_once()
        self.assertFalse((self.root / 'runs').exists())

    def test_missing_local_worker_directory_is_caught_by_preflight(self):
        config = json.loads(self.path.read_text())
        config['machines'][0].update(transport='local', cwd=str(self.root / 'missing'))
        self.path.write_text(json.dumps(config))
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(['--config', str(self.path)]), 1)
        self.assertIn('Cannot enter the configured working directory', errors.getvalue())
        self.assertIn('machines[].cwd', errors.getvalue())
        self.assertFalse((self.root / 'runs').exists())

    def test_per_gpu_settings_and_worker_capacity(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        self.assertEqual(len(controller.active), 3)
        first = next(payload for _, payload in self.transport.starts if payload['env']['CUDA_VISIBLE_DEVICES'] == '0')
        self.assertEqual(first['command'][-2:], ['16', '0'])
        self.assertEqual(first['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(sum(a.gpu['id'] == '0' for a in controller.active), 2)
        other = next(payload for _, payload in self.transport.starts if payload['env']['CUDA_VISIBLE_DEVICES'] == '1')
        self.assertEqual(other['command'][-2:], ['8', '0'])

    def test_busy_gpu_stops_all_its_workers_and_requeues_only_after_confirmation(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        interrupted = [a for a in controller.active if a.gpu['id'] == '0']
        unaffected = next(a for a in controller.active if a.gpu['id'] == '1')
        self.transport.busy.add(('host', '0'))
        controller.tick()
        controller.tick()
        self.assertTrue(all((a.directory / 'cancel').exists() for a in interrupted))
        self.assertFalse((unaffected.directory / 'cancel').exists())
        self.assertEqual(len(self.transport.starts), 3)
        # Even a success concurrent with cancellation must be rerun from scratch.
        for a in interrupted:
            remote.write_json(a.directory / 'status.json', {'state': 'succeeded'})
        remote.write_json(unaffected.directory / 'status.json', {'state': 'succeeded'})
        controller.tick()
        restart = controller.active[0]
        self.assertEqual(restart.gpu['id'], '1')
        self.assertEqual(restart.number, 2)
        self.assertNotEqual(restart.directory, interrupted[0].directory)
        self.assertTrue((restart.directory / 'output').is_dir())

    def test_busy_gpu_recovers_only_after_cancelled_workers_stop(self):
        self.config['machines'][0]['gpus'] = [self.config['machines'][0]['gpus'][0]]
        controller = Controller(self.config, self.transport)
        controller.tick()
        interrupted = list(controller.active)
        self.transport.busy.add(('host', '0'))
        controller.tick()
        controller.tick()
        self.transport.busy.clear()
        # A free slot must not launch work while another worker is stopping.
        remote.write_json(interrupted[0].directory / 'status.json', {'state': 'cancelled'})
        controller.tick()
        self.assertEqual(len(self.transport.starts), 2)
        self.assertEqual(controller.active, [interrupted[1]])
        remote.write_json(interrupted[1].directory / 'status.json', {'state': 'cancelled'})
        controller.tick()
        self.assertEqual(len(self.transport.starts), 4)
        self.assertEqual(len(controller.active), 2)
        self.assertTrue(all(a.gpu['id'] == '0' and not a.cancel_requested for a in controller.active))
        self.assertTrue(all(a.directory not in [old.directory for old in interrupted] for a in controller.active))

    def test_run_waits_when_all_gpus_busy_and_resumes_when_free(self):
        self.config['subjects'] = ['one']
        self.config['machines'][0]['gpus'] = [self.config['machines'][0]['gpus'][0]]
        self.transport.busy.add(('host', '0'))
        controller = Controller(self.config, self.transport)

        def advance(_seconds):
            if self.transport.busy:
                self.assertFalse(self.transport.starts)
                self.transport.busy.clear()
            else:
                self.assertEqual(len(controller.active), 1)
                remote.write_json(controller.active[0].directory / 'status.json', {'state': 'succeeded'})

        with patch('device_jobs.controller.time.sleep', side_effect=advance) as sleep:
            result = controller.run()
        self.assertEqual(result, 0)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(controller.results['one']['state'], 'succeeded')
        self.assertEqual(len(self.transport.starts), 1)

    def test_offline_gpu_does_not_renew_lease_or_duplicate_job(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        heartbeat = controller.active[0].directory / 'heartbeat'
        os.utime(heartbeat, (100, 100))
        self.transport.unreachable.add('host')
        controller.tick()
        self.assertEqual(heartbeat.stat().st_mtime, 100)
        self.assertEqual(len(self.transport.starts), 3)

    def test_uncertain_launch_is_held_until_remote_termination(self):
        self.transport.uncertain = True
        controller = Controller(self.config, self.transport)
        controller.tick()
        controller.tick()
        self.assertEqual(len(self.transport.starts), 3)
        self.assertEqual(len(controller.active), 3)
        self.assertTrue(all(a.cancel_requested for a in controller.active))

    def test_failure_retry_limit_and_terminal_results(self):
        self.config['max_attempts'] = 1
        controller = Controller(self.config, self.transport)
        controller.tick()
        for a in controller.active:
            remote.write_json(a.directory / 'status.json', {'state': 'failed', 'returncode': 2})
        controller.tick()
        self.assertFalse(controller.active)
        self.assertFalse(controller.pending)
        self.assertEqual(len(controller.results), 3)
        self.assertTrue(all(v['state'] == 'failed' for v in controller.results.values()))

    def test_probe_excludes_own_process_group_but_counts_graphics_and_other_work(self):
        directory = self.root / 'attempt'
        directory.mkdir()
        remote.write_json(directory / 'status.json', {'state': 'running', 'pgid': 123})
        rows = [[['0', 'GPU-one', '1000', '650', '75']],
                [['GPU-one', '10', '400'], ['GPU-one', '20', '100']],
                [['0', 'GPU-one', '1000', '650', '75']]]
        with patch.object(remote, 'query', side_effect=rows), patch.object(os, 'getpgid', side_effect=[123, 456]):
            result = remote.probe([str(directory)])
        self.assertEqual(result['0']['foreign_fraction'], .25)
        self.assertEqual(result['0'], result['GPU-one'])

    def test_probe_does_not_count_own_allocation_between_queries_as_foreign(self):
        directory = self.root / 'attempt'
        directory.mkdir()
        remote.write_json(directory / 'status.json', {'state': 'running', 'pgid': 123})
        # Allocation starts after the process sample, or is released before it.
        for before, own, after in [(0, 0, 6400), (6400, 0, 0), (0, 6400, 6400)]:
            with self.subTest(before=before, own=own, after=after):
                rows = [[['0', 'GPU-one', '24000', str(before), '0']],
                        [['GPU-one', '10', str(own)]],
                        [['0', 'GPU-one', '24000', str(after), '0']]]
                with patch.object(remote, 'query', side_effect=rows), patch.object(os, 'getpgid', return_value=123):
                    self.assertEqual(remote.probe([str(directory)])['0']['foreign_mib'], 0)

    def test_transient_busy_reading_keeps_jobs_alive_and_blocks_new_launches(self):
        self.config['machines'][0]['gpus'] = [self.config['machines'][0]['gpus'][0]]
        controller = Controller(self.config, self.transport)
        controller.tick()
        heartbeat = controller.active[0].directory / 'heartbeat'
        os.utime(heartbeat, (100, 100))
        self.transport.busy.add(('host', '0'))
        controller.tick()
        self.assertFalse(any(a.cancel_requested for a in controller.active))
        self.assertGreater(heartbeat.stat().st_mtime, 100)
        self.assertEqual(len(self.transport.starts), 2)
        self.transport.busy.clear()
        controller.tick()
        self.transport.busy.add(('host', '0'))
        controller.tick()
        self.assertFalse(any(a.cancel_requested for a in controller.active))
        controller.tick()
        self.assertTrue(all(a.cancel_requested for a in controller.active))

    def test_interruptions_do_not_exhaust_failure_budget_or_reuse_attempt_dirs(self):
        self.config['subjects'] = ['one']
        self.config['max_attempts'] = 2
        controller = Controller(self.config, self.transport)
        controller.tick()
        first = controller.active[0]
        remote.write_json(first.directory / 'status.json', {'state': 'failed'})
        controller.tick()
        directories = {first.directory}
        for _ in range(4):
            attempt = controller.active[0]
            self.assertNotIn(attempt.directory, directories)
            directories.add(attempt.directory)
            controller.cancel(attempt)
            remote.write_json(attempt.directory / 'status.json', {'state': 'cancelled'})
            controller.tick()
            self.assertNotIn('one', controller.results)
            self.assertEqual(controller.failure_counts['one'], 1)
        remote.write_json(controller.active[0].directory / 'status.json', {'state': 'failed'})
        controller.tick()
        self.assertEqual(controller.results['one']['state'], 'failed')
        self.assertEqual(controller.failure_counts['one'], 2)

    def test_lease_expiry_is_retried_but_command_timeout_counts_as_failure(self):
        self.config['subjects'] = ['one']
        self.config['max_attempts'] = 1
        controller = Controller(self.config, self.transport)
        controller.tick()
        remote.write_json(controller.active[0].directory / 'status.json',
                          {'state': 'cancelled', 'reason': 'controller lease expired'})
        controller.tick()
        self.assertEqual(controller.failure_counts['one'], 0)
        self.assertEqual(controller.active[0].number, 2)
        remote.write_json(controller.active[0].directory / 'status.json',
                          {'state': 'cancelled', 'reason': 'command timeout'})
        controller.tick()
        self.assertEqual(controller.results['one']['state'], 'failed')
        self.assertEqual(controller.failure_counts['one'], 1)

    def test_invalid_config_rejected_before_remote_execution(self):
        for mutate in (lambda c: c['machines'][0]['gpus'][0].update(workers=0),
                       lambda c: c.update(subjects=['../unsafe']),
                       lambda c: c['task'].update(command=['echo', '{subject}']),
                       lambda c: c['task']['params'].pop('batch'),
                       lambda c: c.update(busy_confirmation_polls=0),
                       lambda c: c.update(lease_seconds=10)):
            c = json.loads(self.path.read_text())
            mutate(c)
            path = self.root / 'bad.json'
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_manifest_is_sharded_per_subject(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'dataset': {'kind': 'fixture'},
                                    'subjects': [{'subject_id': s} for s in self.config['subjects']]}))
        self.config['task']['manifest'] = str(path)
        self.config['task']['command'] += ['{manifest}']
        controller = Controller(self.config, self.transport)
        controller.tick()
        for attempt in controller.active:
            shard = json.loads((attempt.directory / 'manifest.json').read_text())
            self.assertEqual(shard['subjects'], [{'subject_id': attempt.subject}])
            self.assertEqual(shard['dataset']['kind'], 'fixture')


    def test_controller_executes_real_detached_helper_and_publishes_successful_outputs(self):
        source = Path(remote.__file__).read_text()
        class LocalTransport(FakeSSH):
            def call(inner, machine, action, payload):
                if action == 'probe':
                    return super().call(machine, action, payload)
                data = dict(payload, _source=source)
                result = subprocess.run([sys.executable, '-c', source, 'start'],
                                        input=json.dumps(data), capture_output=True,
                                        text=True, timeout=5, check=True)
                return json.loads(result.stdout)
        self.config['task']['command'] = [
            sys.executable, '-c',
            'import os,pathlib,sys; pathlib.Path(sys.argv[1], "result").write_text(sys.argv[2] + ":" + os.environ["CUDA_VISIBLE_DEVICES"])',
            '{output_dir}', '{subject}']
        controller = Controller(self.config, LocalTransport())
        deadline = time.monotonic() + 10
        try:
            while (controller.pending or controller.active) and time.monotonic() < deadline:
                controller.tick()
                time.sleep(.05)
            self.assertFalse(controller.active)
            self.assertFalse(controller.pending)
            self.assertEqual(len(controller.results), 3)
            for subject, result in controller.results.items():
                self.assertEqual(result['state'], 'succeeded')
                content = (Path(result['attempt_dir']) / 'output' / 'result').read_text()
                self.assertTrue(content.startswith(subject + ':'))
        finally:
            for attempt in controller.active:
                controller.cancel(attempt)



class RemoteSupervisorTests(unittest.TestCase):
    def test_remote_watchdog_kills_parent_and_child_before_reporting_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'heartbeat').touch()
            child_file = directory / 'child'
            script = ('import subprocess,sys,time,pathlib; '
                      'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                      'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)')
            payload = dict(attempt_dir=tmp, lease_seconds=60, cwd=tmp, env={},
                           command=[sys.executable, '-c', script, str(child_file)])
            helper = subprocess.Popen([sys.executable, '-m', 'device_jobs.remote', 'supervise', json.dumps(payload)])
            try:
                deadline = time.monotonic() + 5
                while not child_file.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(child_file.exists())
                pgid = json.loads((directory / 'status.json').read_text())['pgid']
                (directory / 'cancel').touch()
                helper.wait(timeout=15)
                state = json.loads((directory / 'status.json').read_text())
                self.assertEqual(state['state'], 'cancelled')
                self.assertEqual(remote.group_members(pgid), [])
            finally:
                if helper.poll() is None:
                    helper.kill()
                    helper.wait()

    def test_expired_lease_does_not_launch_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'heartbeat').touch()
            os.utime(directory / 'heartbeat', (100, 100))
            payload = dict(attempt_dir=tmp, lease_seconds=1, cwd=tmp, env={},
                           command=[sys.executable, '-c', 'raise RuntimeError("must not run")'])
            remote.supervise(payload)
            self.assertEqual(json.loads((directory / 'status.json').read_text())['state'], 'cancelled')
            self.assertFalse((directory / 'job.log').exists())

    def test_command_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'heartbeat').touch()
            remote.supervise(dict(attempt_dir=tmp, lease_seconds=60, cwd=tmp, env={},
                                  command=[sys.executable, '-c', 'raise SystemExit(7)']))
            state = json.loads((directory / 'status.json').read_text())
            self.assertEqual(state['state'], 'failed')
            self.assertEqual(state['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
