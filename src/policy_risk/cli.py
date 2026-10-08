from __future__ import annotations

import argparse
import logging

from . import pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="policy-risk")
    parser.add_argument("--config", default="configs/policy.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipeline.run(args.config, args.raw_dir, args.out)


if __name__ == "__main__":
    main()
