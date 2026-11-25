"""Portable, non-shell OpenROAD command execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Mapping, Sequence


class OpenROADNotFound(FileNotFoundError):
    """Raised when the configured OpenROAD executable cannot be resolved."""


class OpenROADExecutionError(RuntimeError):
    """Raised by callers that choose to convert a failed command to an error."""

    def __init__(self, result: "CommandResult") -> None:
        self.result = result
        super().__init__(
            f"OpenROAD exited with status {result.returncode}: "
            f"{' '.join(result.command)}"
        )


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one OpenROAD invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class OpenROADCommandRunner:
    """Run OpenROAD without a shell and retain all process evidence.

    ``executable`` may be a PATH name (the portable default) or an explicit
    executable path supplied by the deployment machine.  The runner does not
    know where OpenROAD-flow-scripts is installed.
    """

    def __init__(self, executable: str = "openroad", *, threads: int | None = None) -> None:
        if threads is not None and threads <= 0:
            raise ValueError("threads must be positive")
        self.executable = executable
        self.threads = threads

    def resolve(self) -> str:
        candidate = Path(self.executable)
        if candidate.parent != Path("."):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            raise OpenROADNotFound(
                f"OpenROAD executable is not executable: {self.executable!r}"
            )
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise OpenROADNotFound(
                f"OpenROAD executable not found on PATH: {self.executable!r}"
            )
        return resolved

    def run(
        self,
        *,
        script: Path | str | None = None,
        commands: Sequence[str] = (),
        cwd: Path | str | None = None,
        timeout_s: float = 1800.0,
        environment: Mapping[str, str] | None = None,
        log_dir: Path | str | None = None,
    ) -> CommandResult:
        """Execute one Tcl file or an in-memory sequence of Tcl commands.

        The generated command file is removed after the process exits.  When
        ``log_dir`` is supplied, stdout and stderr are copied to that directory
        for lightweight experiment logging.
        """

        if script is not None and commands:
            raise ValueError("specify either script or commands, not both")
        if script is None and not commands:
            raise ValueError("at least one script or command is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        executable = self.resolve()
        working_directory = Path(cwd).resolve() if cwd is not None else None
        if working_directory is not None and not working_directory.is_dir():
            raise NotADirectoryError(working_directory)

        temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        script_path: Path
        if script is not None:
            script_path = Path(script).resolve()
            if not script_path.is_file():
                raise FileNotFoundError(script_path)
        else:
            temporary_directory = tempfile.TemporaryDirectory(prefix="mappo-eda-")
            script_path = Path(temporary_directory.name) / "commands.tcl"
            script_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

        command = (executable, "-no_init", "-exit", *(("-threads", str(self.threads)) if self.threads is not None else ()), str(script_path))
        process_environment = os.environ.copy()
        if environment is not None:
            process_environment.update({str(k): str(v) for k, v in environment.items()})

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=working_directory,
                env=process_environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=stdout,
                stderr=stderr + "\nOpenROAD command timed out",
                duration_s=time.monotonic() - started,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()
        if "result" not in locals():
            result = CommandResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_s=time.monotonic() - started,
            )

        if log_dir is not None:
            output_directory = Path(log_dir)
            output_directory.mkdir(parents=True, exist_ok=True)
            (output_directory / "openroad.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (output_directory / "openroad.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
        return result


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
