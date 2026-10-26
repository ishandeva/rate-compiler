"""
Rate Parser backward compatibility wrapper.
The module has been renamed to RateCompiler (rate_compiler.py) to differentiate
it from the Ingestion Service's Excel extraction parser.
"""

from rate_compiler import (
    RateCompiler,
    FeeRateCompiler,
    FeeRateParser,
    RateCompilerError,
    RateParserError,
    SemanticClassificationError,
    InvariantViolationError,
    TierBracket,
    RawTierBracket,
    RawParsedSchedule,
    NormalizedFeeRule,
    LexicalSurfaceCleaner,
    SemanticClassifier,
    InvariantVerification,
    BENCHMARK_CASES,
    ADDITIONAL_BENCHMARK_CASES,
    run_benchmarks,
)

__all__ = [
    "RateCompiler",
    "FeeRateCompiler",
    "FeeRateParser",
    "RateCompilerError",
    "RateParserError",
    "SemanticClassificationError",
    "InvariantViolationError",
    "TierBracket",
    "RawTierBracket",
    "RawParsedSchedule",
    "NormalizedFeeRule",
    "LexicalSurfaceCleaner",
    "SemanticClassifier",
    "InvariantVerification",
    "BENCHMARK_CASES",
    "ADDITIONAL_BENCHMARK_CASES",
    "run_benchmarks",
]

if __name__ == "__main__":
    import sys
    from rate_compiler import run_benchmarks
    # Pass through CLI execution
    import argparse
    parser = argparse.ArgumentParser(description="Rate Compiler 2.0 (Compatibility Mode)")
    parser.add_argument("--format", "-f", default="json", choices=["json", "compact_json", "dsl", "ml", "summary"])
    parser.add_argument("--cases", "-c", default="all", choices=["all", "additional", "benchmark"])
    args = parser.parse_args()

    selected = []
    if args.cases in ("all", "benchmark"):
        selected.extend(BENCHMARK_CASES)
    if args.cases in ("all", "additional"):
        selected.extend(ADDITIONAL_BENCHMARK_CASES)

    print(run_benchmarks(selected, output_type=args.format))
