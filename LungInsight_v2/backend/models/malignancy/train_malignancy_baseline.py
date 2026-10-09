#!/usr/bin/env python3
"""Train the score-only baseline using the shared malignancy training loop."""
from __future__ import annotations

from train_malignancy import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args(), input_dim=6))