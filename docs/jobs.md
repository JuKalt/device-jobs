# Job configuration

A schema version 1 JSON document selects job IDs in `subjects`, sets
`shared_run_root`, and defines one `task` and one or more `machines`.
The term subject is retained in the interface for compatibility. It is an
arbitrary safe job ID. Commands are argument arrays. Use `{python}`,
`{subject}`, `{output_dir}`, `{attempt_dir}`, and optionally `{manifest}`
in a command. The job sees its assigned GPU through `CUDA_VISIBLE_DEVICES`.
When a program requires a device argument, use its logical device 0.
Arguments and environment values can also use `task.params`, overridden by
machine then GPU parameters. Define `task.manifest` to shard a JSON manifest
with a `subjects` list and matching `subject_id` values.

Use a stable shared `output_root` for resumable jobs. By default
`{output_dir}` is a per-job directory. A processor whose own command appends
the subject ID uses `task.output_layout: "root"`. A file or symlink in place
of a job output directory is rejected. One controller holds an advisory lock
on each shared output root. Workers and controller need the same absolute
paths for shared outputs, logs and input manifests.

## Worker project

`machines[].cwd` points at the **job project's** working directory on a worker.
The scheduler can run a checkout already prepared on that worker. To synchronize
selected files, add:
```json
"source": {
  "root": "/path/to/controller/job-project",
  "include": ["pyproject.toml", "example_job/*"]
}
```

The source root is a Git working tree. The controller snapshots the current
contents of tracked files matching the explicit patterns and transfers changes
with checksums. `machines[].cwd` must already be a dedicated Git working tree on
each worker. A new one can be initialized with `git init`. Paths must be regular,
non-hidden files and must stay inside that checkout. The scheduler removes
obsolete files from its previous managed manifest only when their contents
still match the recorded files. Model assets and credentials are set up
separately. `source` replaces the combined repo's `sync_repository` flag.

## Transport and recovery

Use `transport: "local"` for the controller host or `"ssh"` with an existing
trusted host key and noninteractive key or agent authentication. Provide
`bootstrap_python` for the system helper and `python` for the task. A machine
or the top-level config can define `provision` entries containing `name`,
`check` and `setup`. Each has an argument array for existing runtime checks or
installation. Use `--dry-run` to inspect commands and source selection.

Before a live batch, Device Jobs checks SSH where required, `bootstrap_python`,
and `cwd` on every worker. If any check fails, the batch stops before creating a run or
starting jobs, and the terminal lists the failed workers. Run
`./run.sh --config /path/to/job.json --preflight-only` to repeat just this check.
It does not run provisioning or check GPUs. `--dry-run` remains offline.

If the worker rejects a passphrase-protected key, load the key into an agent
from the same terminal that will launch Device Jobs. Start an agent only if
that terminal does not already have one:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Use the key selected by `identity_file` or your SSH configuration if it differs.
Enter the passphrase at the `ssh-add` prompt. Device Jobs does not collect
credentials. For host key errors, verify the host fingerprint before updating
`known_hosts`.

Treat timeouts and retries as part of the workload contract. The scheduler
retains completed outputs and attempts. A job must either resume safely or use
attempt-specific scratch space. Exit code 75 requests a retry without counting
as a job failure. `max_attempts` and `max_interruptions` cap retries.
`task.timeout_seconds` caps individual attempts. Unknown GPU usage, including
`[N/A]`, blocks new jobs on that GPU and lets existing jobs retain their leases.
Check the shared run folder for queue state, logs and terminal results.
