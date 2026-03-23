#!/usr/bin/env python3
"""Small Python wrapper around the `treba` CLI binary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


@dataclass
class TrebaResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


class TrebaRunner:
    """Executes Treba commands and returns captured output."""

    def __init__(self, treba_bin: str = "treba") -> None:
        self.treba_bin = self._resolve_bin(treba_bin)

    @staticmethod
    def _resolve_bin(treba_bin: str) -> str:
        if Path(treba_bin).exists():
            return str(Path(treba_bin).resolve())
        if treba_bin == "treba" and Path("./treba").exists():
            return str(Path("./treba").resolve())
        resolved = shutil.which(treba_bin)
        if resolved is None:
            raise FileNotFoundError(f"Could not find Treba binary: {treba_bin}")
        return resolved

    def run(self, args: Sequence[str]) -> TrebaResult:
        cmd = [self.treba_bin, *args]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return TrebaResult(
            cmd=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
