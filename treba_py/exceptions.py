class TrebaError(Exception):
    """Base exception for the draft Python Treba wrapper."""


class TrebaCommandError(TrebaError):
    """Raised when the `treba` process exits with a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        args: list[str] | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.command_args = args or []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class AlphabetError(TrebaError):
    """Raised for token/alphabet mapping inconsistencies."""


class NotFittedError(TrebaError):
    """Raised when inference methods are called before `fit()`."""
