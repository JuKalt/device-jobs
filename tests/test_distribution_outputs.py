"""Stable subject outputs with separate logs and confirmed retry preservation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch

from device_jobs.controller import Controller, load_config, render
from device_jobs import remote
from tests.test_distribution import FakeSSH


class DistributionOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = dict(schema_version=1, sync_repository=False, subjects=['one'], shared_run_root=str(self.root / 'logs'),
                      output_root=str(self.root / 'processed'),
                      task={'output_layout': 'root', 'command': ['python3', 'task.py', '{subject}', '{output_dir}']},
                      machines=[{'name': 'machine', 'host': 'machine', 'cwd': str(self.root),
                                 'gpus': [{'id': 0}, {'id': 1}]}])
        self.path = self.root / 'config.json'
        self.path.write_text(json.dumps(config))
        self.config = load_config(self.path)
        self.transport = FakeSSH()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_root_layout_matches_normal_converter_and_preserves_existing_output_on_first_attempt(self):
        output = Path(self.config['output_root']) / 'one'
        output.mkdir(parents=True)
        existing = output / 'completed-images'
        existing.touch()
        controller = Controller(self.config, self.transport)
        controller.tick()
        self.assertEqual(self.transport.starts[0][1]['command'][-1], self.config['output_root'])
        self.assertTrue(existing.exists())
        self.assertFalse((controller.active[0].directory / 'output').exists())
        self.assertTrue((controller.active[0].directory / 'job.json').exists())

    def test_partial_output_preserved_after_termination_before_retry(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        attempt = controller.active[0]
        output = Path(self.config['output_root']) / 'one'
        output.mkdir()
        (output / 'partial').touch()
        other = Path(self.config['output_root']) / 'other'
        other.mkdir()
        (other / 'done').touch()
        self.transport.busy.add(('machine', '0'))
        controller.tick()
        controller.tick()
        self.assertTrue((output / 'partial').exists())
        self.assertEqual(len(self.transport.starts), 1)
        remote.write_json(attempt.directory / 'status.json', {'state': 'cancelled'})
        controller.tick()
        self.assertTrue((output / "partial").exists())
        self.assertTrue((other / 'done').exists())
        self.assertEqual(controller.active[0].number, 2)
        self.assertEqual(controller.active[0].gpu['id'], '1')
        self.assertEqual(self.transport.starts[1][1]['command'][-1], self.config['output_root'])
        self.assertTrue(attempt.directory.exists())  # logs and resumable output retained

    def test_success_keeps_flat_output_and_records_it_in_results(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        attempt = controller.active[0]
        output = Path(self.config['output_root']) / 'one'
        output.mkdir()
        (output / 'smpl_params.npz').touch()
        remote.write_json(attempt.directory / 'status.json', {'state': 'succeeded'})
        controller.tick()
        self.assertTrue((output / 'smpl_params.npz').exists())
        self.assertEqual(controller.results['one']['output_dir'], str(output))

    def test_final_failed_attempt_preserves_resumable_output(self):
        self.config['max_attempts'] = 1
        controller = Controller(self.config, self.transport)
        controller.tick()
        output = Path(self.config['output_root']) / 'one'
        output.mkdir()
        (output / 'partial').touch()
        remote.write_json(controller.active[0].directory / 'status.json', {'state': 'failed'})
        controller.tick()
        self.assertTrue((output / "partial").exists())
        self.assertEqual(controller.results['one']['state'], 'failed')

    def test_lease_renewal_continues_while_tick_is_blocked(self):
        controller = Controller(self.config, self.transport)
        controller.config['poll_seconds'] = 0
        controller.config['lease_seconds'] = .08
        original_tick = controller.tick
        started = []
        def tick():
            if not started:
                original_tick()
                attempt = controller.active[0]
                started.append(attempt)
                initial = (attempt.directory / 'heartbeat').stat().st_mtime_ns
                # Exceed the lease while the main controller is blocked.
                time.sleep(.12)
                self.assertGreater((attempt.directory / 'heartbeat').stat().st_mtime_ns, initial)
                remote.write_json(attempt.directory / 'status.json', {'state': 'succeeded'})
            else:
                original_tick()
        with patch.object(controller, 'tick', side_effect=tick):
            self.assertEqual(controller.run(), 0)
        self.assertFalse((started[0].directory / 'cancel').exists())

    def test_subject_layout_is_stable_across_attempt_log_directories(self):
        self.config['task']['output_layout'] = 'subject'
        machine = self.config['machines'][0]
        one, _ = render(self.config, machine, machine['gpus'][0], 'one', Path('/logs/attempt-1'))
        two, _ = render(self.config, machine, machine['gpus'][0], 'one', Path('/logs/attempt-2'))
        self.assertEqual(one[-1], str(Path(self.config['output_root']) / 'one'))
        self.assertEqual(one, two)

    def test_overlap_between_logs_and_outputs_rejected(self):
        config = json.loads(self.path.read_text())
        config['output_root'] = config['shared_run_root'] + '/output'
        self.path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_resume_never_follows_subject_symlink(self):
        controller = Controller(self.config, self.transport)
        controller.tick()
        attempt = controller.active[0]
        target = self.root / 'input'
        target.mkdir()
        (target / 'keep').touch()
        (Path(self.config['output_root']) / 'one').symlink_to(target)
        remote.write_json(attempt.directory / 'status.json', {'state': 'failed'})
        controller.collect()
        self.assertEqual(controller.results['one']['state'], 'failed')
        self.assertIn('symlink', controller.results['one']['reason'])
        self.assertNotIn('one', controller.pending)
        self.assertTrue((target / 'keep').exists())


if __name__ == '__main__':
    unittest.main()
