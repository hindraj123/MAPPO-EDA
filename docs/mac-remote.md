# View desktop training from a Mac

The Mac needs Python 3.10+, SSH, and (for opening layouts) KLayout. Training
and real checkpoint evaluation run on the Ubuntu desktop. No PyTorch, Docker,
OpenROAD, or Yosys installation is required for this remote viewer.

## Configure once

From your Mac's checkout, create the ignored `configs/remote.local.json`:

```bash
python3 scripts/remote_training.py init \
  --host YOUR_USER@YOUR_DESKTOP \
  --repo /absolute/desktop/path/to/MAPPO-EDA \
  --run-dir artifacts/YOUR_RUN
```

Alternatively copy `configs/remote.example.json` to `configs/remote.local.json`
and edit it. Use an SSH alias in `host` for custom ports or keys configured in
`~/.ssh/config`. Authentication stays with SSH; do not put passwords or tokens
in the JSON. The desktop must be awake and reachable from the Mac.

Both checkouts need `scripts/training_dashboard.py`. The viewer reads existing
run artifacts; it does not launch training or evaluation. Checkpoint results
require a completed `evaluation.json` produced by the desktop evaluator.

## Use

```bash
python3 scripts/remote_training.py dashboard
python3 scripts/remote_training.py eval
python3 scripts/remote_training.py download --open
```

`dashboard` shows training and evaluation together. `eval` shows the checkpoint
evaluation only. Add `--once` to either for a static snapshot. Ctrl+C closes the
viewer without stopping desktop training. Time remaining is an estimate based
on update completions observed during the viewer session.

`download` retrieves the GDS from the **last episode of the completed real
evaluation**, together with its JSON report (all episodes and tool versions),
into `~/Downloads/MAPPO-EDA`. It does not substitute the training run's best GDS.
Repeating the command refreshes files for the same checkpoint. Downloads are
explicit; the command does not watch for or evaluate future checkpoints.

To open a previously downloaded layout without reconnecting:

```bash
python3 scripts/remote_training.py open
```

Install KLayout from https://www.klayout.de/build.html if needed. The client
does not bypass macOS security controls. If macOS blocks the app, follow Apple's
instructions at https://support.apple.com/102445. Layout inspection is local;
re-running timing or DRC requires the matching technology and verification setup.
