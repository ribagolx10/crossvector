"""
CrossVector Benchmark Suite

Tools for comprehensive benchmarking of vector database adapters.

Usage:
    # Run benchmarks
    python -m scripts.benchmark [options]

    # Or directly
    python scripts/benchmark/run.py [options]

    # Generate fixtures
    python scripts/benchmark/generate_fixtures.py [options]
"""

from .generate_fixtures import (
    generate_benchmark_docs,
    generate_nested_metadata,
    generate_realistic_text,
    generate_search_queries,
)
from .run import BenchmarkRunner, main

__all__ = [
    # Fixture generators
    "generate_benchmark_docs",
    "generate_search_queries",
    "generate_nested_metadata",
    "generate_nested_metadata",
    "generate_realistic_text",
    # Benchmark runner
    "BenchmarkRunner",
    "main",
]


# CLI entry point
if __name__ == "__main__":
    main()
