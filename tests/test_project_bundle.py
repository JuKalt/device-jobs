import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from device_jobs.remote import repository_probe, repository_update
from device_jobs.repository import snapshot


class SourceBundleTests(unittest.TestCase):
    def test_selected_tracked_files_round_trip_to_an_independent_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            controller = root / "controller"
            for checkout in (worker, controller):
                subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            package = controller / "toy_job"
            package.mkdir()
            (package / "run.py").write_text("print('hello')\n")
            (controller / "personal-notes.txt").write_text("excluded\n")
            (controller / "untracked.txt").write_text("excluded\n")
            subprocess.run(["git", "-C", str(controller), "add", "toy_job", "personal-notes.txt"], check=True)
            bundle = snapshot(controller, ["toy_job/*"])
            self.assertEqual(set(bundle["manifest"]), {"toy_job/run.py"})
            request = {"cwd": str(worker), **bundle}
            probe = repository_probe(request)
            self.assertEqual(probe["changed"], ["toy_job/run.py"])
            updated = repository_update({**request, **probe, "files": bundle["files"]})
            self.assertEqual(updated["updated"], 1)
            self.assertEqual(repository_probe(request), {"changed": [], "obsolete": []})
            result = subprocess.run(["python3", str(worker / "toy_job/run.py")],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, "hello\n")
            (worker / "toy_job/run.py").write_text("# local edits")
            with self.assertRaisesRegex(ValueError, "locally modified"):
                repository_update({"cwd": str(worker), "manifest": {}, "digest": "",
                                   "obsolete": ["toy_job/run.py"], "files": {}})

    def test_source_bundle_will_not_follow_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            os.symlink("/etc/hosts", root / "job.py")
            subprocess.run(["git", "-C", str(root), "add", "job.py"], check=True)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                snapshot(root, ["job.py"])
