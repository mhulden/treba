from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Mapping, Sequence

from .exceptions import TrebaCommandError


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float


class TrebaRunner:
    """Thin subprocess backend around the `treba` executable."""

    def __init__(
        self,
        treba_bin: str = "treba",
        *,
        work_dir: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.treba_bin = treba_bin
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.env = dict(env) if env is not None else None
        self._resolved_bin: str | None = None

        candidate = Path(treba_bin)
        if candidate.exists():
            self._resolved_bin = str(candidate.resolve())

    def run(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        if self._resolved_bin is None:
            self._resolved_bin = self._resolve_bin(self.treba_bin, self.work_dir)
        cmd = [self._resolved_bin, *args]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.work_dir,
            env=self.env,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start

        result = CommandResult(
            args=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_sec=elapsed,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            suffix = f" | {detail}" if detail else ""
            raise TrebaCommandError(
                f"Treba command failed (exit={result.returncode}): {' '.join(result.args)}{suffix}",
                args=result.args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    @staticmethod
    def _resolve_bin(
        treba_bin: str,
        work_dir: Path | None = None,
    ) -> str:
        # If caller provided an explicit path, prefer it directly.
        candidate = Path(treba_bin)
        if candidate.exists():
            return str(candidate.resolve())

        # For the default binary name, auto-discover local builds.
        if treba_bin == "treba":
            search_roots: list[Path] = []
            if work_dir is not None:
                search_roots.append(work_dir)
            search_roots.append(Path.cwd())
            # Also try the package root (repo layout: <root>/treba_py/runner.py).
            search_roots.append(Path(__file__).resolve().parent.parent)

            checked: set[Path] = set()
            for root in search_roots:
                for current in [root, *root.parents]:
                    if current in checked:
                        continue
                    checked.add(current)
                    local = current / "treba"
                    if local.exists() and local.is_file():
                        return str(local.resolve())

        resolved = shutil.which(treba_bin)
        if resolved is None:
            raise FileNotFoundError(f"Could not find Treba binary: {treba_bin}")
        return resolved
