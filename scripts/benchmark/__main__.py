"""
CLI entry point for benchmark module.

Allows running benchmarks via:
    python -m scripts.benchmark [options]
"""

from .run import main

if __name__ == "__main__":
    main()
