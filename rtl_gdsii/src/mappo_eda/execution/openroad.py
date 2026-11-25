"""Small, testable wrapper around the OpenROAD executable."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class OpenROADNotFound(FileNotFoundError):
    """Raised when the configured OpenROAD executable is unavailable."""


class OpenROADRunner:
    def __init__(self, executable: str = "openroad") -> None:
        self.executable = executable

    def resolve(self) -> str:
        path = shutil.which(self.executable)
        if path is None:
            raise OpenROADNotFound(
                f"OpenROAD executable not found: {self.executable!r}. "
                "Install OpenROAD or pass an absolute executable path."
            )
        return path

    def run(
        self,
        *,
        script: Path | None = None,
        commands: Sequence[str] = (),
        cwd: Path | None = None,
        timeout_s: float = 300.0,
    ) -> CommandResult:
        if script is not None and commands:
            raise ValueError("Specify either script or commands, not both")
        if script is None and not commands:
            raise ValueError("At least one script or command is required")
        args = [self.resolve(), "-exit"]
        temporary_script: str | None = None
        if script is not None:
            args.extend(["-file", str(script)])
        elif commands:
            handle = tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False, encoding="utf-8")
            temporary_script = handle.name
            with handle:
                handle.write("\n".join(commands) + "\n")
            args.extend(["-no_init", "-file", temporary_script])
        try:
            completed = subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        finally:
            if temporary_script is not None:
                Path(temporary_script).unlink(missing_ok=True)
        return CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)
