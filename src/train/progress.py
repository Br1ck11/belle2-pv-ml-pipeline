"""Live progress within an epoch (or a scoring pass), enabled by [run] verbose = 1.

In a terminal the line is rewritten in place (carriage return), at most every
0.2 s. When stdout is not a terminal (redirected to a log file, a notebook,
a job system), a normal line is printed at every 10 % instead, so logs stay
readable. verbose = 0 prints nothing here; the one-line epoch summaries of
the training script are printed either way.
"""

import sys
import time


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class BatchProgress:
    """progress = BatchProgress("epoch 3/20 train", n_batches); progress.update(k, windows, loss); progress.close()"""

    def __init__(self, label: str, total: int, *, stream=None, width: int = 24) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.stream = stream or sys.stdout
        self.width = width
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = time.perf_counter()
        self.last_draw = 0.0
        self.next_percent = 10
        self.line_length = 0

    def update(self, done: int, windows: int, loss: float | None = None) -> None:
        now = time.perf_counter()
        elapsed = now - self.started
        percent = 100.0 * done / self.total
        finished = done >= self.total
        if self.interactive:
            if not finished and now - self.last_draw < 0.2:
                return
        elif percent < self.next_percent and not finished:
            return
        self.last_draw = now
        while self.next_percent <= percent:
            self.next_percent += 10
        rate = windows / elapsed if elapsed > 0 else 0.0
        remaining = elapsed * (self.total - done) / done if done else 0.0
        filled = int(self.width * done / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        text = (f"{self.label} [{bar}] {percent:5.1f}%  batch {done}/{self.total}"
                + (f"  loss {loss:.5g}" if loss is not None else "")
                + f"  {rate:,.0f} windows/s  {_clock(elapsed)} elapsed, ~{_clock(remaining)} left")
        if self.interactive:
            padding = " " * max(0, self.line_length - len(text))
            self.stream.write("\r" + text + padding)
            self.line_length = len(text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def close(self) -> None:
        """End the in-place line so the next message starts on a fresh line."""
        if self.interactive and self.line_length:
            self.stream.write("\r" + " " * self.line_length + "\r")
            self.stream.flush()
            self.line_length = 0
