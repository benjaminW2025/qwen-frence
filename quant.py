#!/usr/bin/env python3
"""Single entry point for the quantization experiments.

  python quant.py full-model --help
  python quant.py attention --help
  python quant.py capture --help
  python quant.py baselines --help
  python quant.py test
"""
import argparse
import importlib
import sys
import unittest
from pathlib import Path


def main(argv=None):
    commands = {"full-model": "full_model", "attention": "attention", "capture": "capture",
                "baselines": "suite"}
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in commands:
        module = importlib.import_module("experiments.quant." + commands[argv[0]])
        return module.main(argv[1:]) or 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*commands, "test"])
    args, remaining = parser.parse_known_args(argv)
    if args.command == "test":
        if remaining:
            parser.error("test takes no arguments")
        suite = unittest.defaultTestLoader.discover(
            str(Path(__file__).resolve().parent / "experiments/quant/tests"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
