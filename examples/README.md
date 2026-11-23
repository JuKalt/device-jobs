# Device Jobs examples

From an installed VCI Suite checkout, run `./device-jobs/run.sh --help` at
its root. To follow the examples below, run `cd device-jobs` once. Skip the
standalone install if you already installed the suite. For a standalone
checkout, run `./install.sh` once from the Device Jobs root.

All examples use `./run.sh`, with no environment activation. The launcher
uses the installed scheduler if available or runs the source with Python 3.10+.
With a Conda install, the launcher uses its recorded environment and leaves
your current environment active in the terminal.
A dry run prints the command without launching it. Live jobs use the NVIDIA
GPU scheduler, even when the hello script itself uses no GPU.

## Process a VCI capture batch across machines

This is the Device Jobs equivalent of the combined repository's command:

```bash
python3 distribute_subjects.py --config configs/distribution/vci-2026_09_17.json
```

From the **Device Jobs checkout** (run `cd device-jobs` from VCI Suite), copy the complete
[VCI batch config](vci-2026_09_17.json) to a file you can edit:

```bash
mkdir -p configs/distribution
cp examples/vci-2026_09_17.json configs/distribution/vci-2026_09_17.json
```

The sample schedules one recording per job using Capture Processing's
`process_vci_subject.py`. It shows a GPU on the controller and two GPUs on one
SSH worker. The hostnames, paths and subject IDs are placeholders. Edit them
before launching:

| Set | What it controls |
| --- | --- |
| `subjects` and `task.params.raw_root` | Use the folder names under your raw capture root. Start with one recording. Each eligible worker must be able to read it. |
| `shared_run_root` and `output_root` | Set separate writable absolute paths visible at the same locations on the controller and workers. Logs and leases go under the run root; processed recordings go under `output_root/<subject>`. |
| `machines[].host`, `user`, `transport`, and `cwd` | Set SSH details and the absolute **Capture Processing** checkout path on each machine. Drop the `local` entry if the controller has no GPU. Set up noninteractive SSH via your key or agent. |
| `machines[].bootstrap_python`, `python`, and `params` | Set the system Python for the scheduler helper, the converter interpreter, the Conda prefix and the SAM3D interpreter on each machine. Override `mamma_python` if you use a different MAMMA environment. |
| `machines[].gpus` | List the actual GPU indices or UUIDs from `nvidia-smi`. Each `workers: 1` allows one subject at a time on that GPU. `task.params` sets the per-subject processing defaults; machine and GPU entries can override them. |

Run the full Capture Processing setup on every worker first, including model
access. The example uses prepared checkouts, so remove `sync_repository` when
migrating the old config. If you want the scheduler to synchronize specific
tracked source files instead, configure `source.root` and `source.include`
explicitly. Environments, assets and credentials still need to be set up on
each worker. See the [config reference](../docs/jobs.md#worker-project) for
source sync and [provisioning](../docs/jobs.md#transport-and-recovery) for
optional noninteractive `check` and `setup` steps.

```bash
./run.sh --config configs/distribution/vci-2026_09_17.json --dry-run
./run.sh --config configs/distribution/vci-2026_09_17.json --preflight-only
./run.sh --config configs/distribution/vci-2026_09_17.json
```

The preview renders each GPU's command for the first subject. The preflight
checks login, the Python helper, and the configured working directory without
running jobs. The live command repeats the preflight, then runs the enabled
Capture Processing stages and passes the common output root with
`output_layout: "root"`, so the wrapper appends the subject ID once. Rerun the
same command and paths to resume. The example uses `--resume-validation full`;
use `quick` as in the old config only when source recordings are immutable.

## Run a local Python script on one GPU slot

```bash
cp examples/python-job.json /tmp/device-jobs-local.json
```

Edit the copy: point `machines[0].cwd` to this checkout, set its GPU ID and
interpreter, and make `shared_run_root` and `output_root` two separate writable
absolute paths. Set `subjects` to `["sample-a", "sample-b"]`. Then run:

```bash
./run.sh --config /tmp/device-jobs-local.json --dry-run
./run.sh --config /tmp/device-jobs-local.json --preflight-only
./run.sh --config /tmp/device-jobs-local.json
```

[hello.py](hello.py) accepts the job ID and output directory as positional
arguments. Expect `sample-a/hello.txt` and `sample-b/hello.txt` under the output
root. See the shared run root for queue status and logs. Run the same spec again
to resume; use different run and output roots for a separate experiment.

## Send selected Python source to an SSH worker

Start from [ssh-job.json](ssh-job.json). Set `source.root` to the controller's
checkout of this repo, and replace the example hostname and user. Set `cwd`
to a dedicated Git working tree on the worker. The scheduler syncs selected
files into that tree; it does not create it. For an empty worker checkout, run
`git init /path/to/hello-job` on the worker before launching. Both hosts must
see the same absolute shared run and output paths. Verify noninteractive SSH access and the worker's Python,
GPU index and `nvidia-smi` before launching. Do not add credentials to the JSON.

```bash
cp examples/ssh-job.json /tmp/device-jobs-ssh.json
# Edit /tmp/device-jobs-ssh.json as described above.
./run.sh --config /tmp/device-jobs-ssh.json --dry-run
./run.sh --config /tmp/device-jobs-ssh.json --preflight-only
```

The preview lists the source snapshot and commands. Preflight checks the
connection, Python helper, and working directory. When both pass, run without
either flag; the worker will receive the tracked `examples/hello.py` file and run it.
If the worker already has a prepared checkout, omit the `source` section and
point `cwd` at that checkout instead. To run your own program, replace the
argument array and keep a `{subject}` or `{manifest}` input plus `{output_dir}`.
Use `{attempt_dir}` when temporary attempt files must be isolated. A task that
writes resumable state should keep it under its stable output directory.

The [job configuration reference](../docs/jobs.md) explains provisioning,
shared filesystems, task parameters, retries and GPU assignment.
