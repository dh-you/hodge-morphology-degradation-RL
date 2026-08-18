"""Fail closed unless scratch has campaign headroom and home is unused."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

MIN_SOFT_HEADROOM_BYTES = 5 * 1024 ** 3


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-root", required=True)
  args = parser.parse_args()
  output = Path(args.output_root).resolve()
  if output == Path.home().resolve() or output.is_relative_to(Path.home().resolve()):
    raise SystemExit("campaign output must not use home quota")
  if not output.is_relative_to(Path("/scratch/network/dy0130")):
    raise SystemExit("campaign output must be under /scratch/network/dy0130")
  text = subprocess.check_output(("quota", "-v"), text=True)
  match = re.search(
      r"172\.28\.62\.2:/scratch/network\s+(\d+)\s+(\d+)\s+(\d+)",
      text,
  )
  if match is None:
    raise SystemExit("could not resolve scratch quota")
  used_kib, soft_kib, hard_kib = map(int, match.groups())
  soft_headroom = (soft_kib - used_kib) * 1024
  hard_headroom = (hard_kib - used_kib) * 1024
  if soft_headroom < MIN_SOFT_HEADROOM_BYTES:
    raise SystemExit(
        f"scratch soft-quota headroom {soft_headroom / 1024**3:.2f} GiB < 5 GiB"
    )
  print(
      f"scratch quota preflight: used={used_kib/1024**2:.2f} GiB "
      f"soft_headroom={soft_headroom/1024**3:.2f} GiB "
      f"hard_headroom={hard_headroom/1024**3:.2f} GiB"
  )


if __name__ == "__main__":
  main()
