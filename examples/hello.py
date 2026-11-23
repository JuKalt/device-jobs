"""A job whose only output is a small text file."""

from pathlib import Path
import sys

subject, output_dir = sys.argv[1:]
output = Path(output_dir)
output.mkdir(parents=True, exist_ok=True)
(output / "hello.txt").write_text(f"Hello, {subject}!\n")
