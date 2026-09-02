"""Export local AutoTuner benchmark evidence for the public GitHub Pages site."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parent
_DEFAULT_DESTINATION = _REPOSITORY_ROOT / "benchmark-site" / "index.html"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a path-redacted, dependency-free benchmark dashboard from "
            "the local AutoTuner settings database."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_DEFAULT_DESTINATION,
        help="HTML destination (default: benchmark-site/index.html)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Read another AutoTuner data directory instead of ~/.autotuner",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty dashboard instead of refusing when no evidence exists",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.data_dir is not None:
        os.environ["AUTOTUNER_DATA_DIR"] = str(args.data_dir.expanduser().resolve())

    # Import only after the optional data-dir override has been applied.
    import app_settings
    from performance_report import write_public_performance_report

    records_by_test = app_settings.list_performance_run_results()
    run_count = sum(len(records) for records in records_by_test.values())
    if run_count == 0 and not args.allow_empty:
        parser.error(
            "no saved benchmark evidence found; run a Performance test first or "
            "pass --data-dir/--allow-empty"
        )

    destination = args.destination.expanduser().resolve()
    write_public_performance_report(records_by_test, destination)
    model_count = len(
        {
            str(record.get("model_name") or record.get("model_path") or "").casefold()
            for records in records_by_test.values()
            for record in records
        }
    )
    print(
        f"Published {run_count} saved runs across {model_count} model(s) to "
        f"{destination} ({destination.stat().st_size:,} bytes)."
    )
    print("Local model/runtime paths were removed and the static page was validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
