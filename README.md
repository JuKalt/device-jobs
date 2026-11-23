# Device Jobs

[Usage examples](examples/README.md): [process a VCI capture batch across machines](examples/README.md#process-a-vci-capture-batch-across-machines), run a local Python job, or send source files to an SSH worker.

Run independent Python jobs across Linux machines using local or SSH transport.
The controller assigns one visible NVIDIA GPU per job, with configurable slots
per GPU. It handles logs, leases, retries, interruptions and output ownership.
Every worker needs a Python bootstrap interpreter, `nvidia-smi`, and access to
the shared run and output paths. Remote workers also need working
noninteractive SSH from the controller. Workload dependencies are separate.

For standalone setup, put Conda on `PATH`. The installer uses `vci-device-jobs`
by default; set `VCI_CLI_ENV` to choose another environment. If you installed
through VCI Suite, skip the installer and use the component's `run.sh` directly.

```bash
./install.sh
./run.sh --config examples/python-job.json --dry-run
```

From the suite root, run `./device-jobs/run.sh`. The launcher can also run
the source directly with Python 3.10+ if this checkout has no installed
scheduler.

To launch, edit the example's paths and GPU index for your environment.
Use `--preflight-only` to check worker access, Python and `cwd` before starting
a batch. A live run repeats those checks. Use a dedicated writable run
directory and output root.
Read [the config contract](docs/jobs.md) for independent checkouts, SSH,
provisioning, and resume rules. Capture Processing and Avatar Creation each
ship a job example using this same interface.
