#!/usr/bin/env python3
"""
VecDB — single entry point.

Usage:
    python run.py benchmark    # run full benchmark + generate plot
    python run.py demo         # interactive search demo
    python run.py all          # benchmark, then demo
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def cmd_benchmark():
    from benchmarks.benchmark import run_benchmark
    from benchmarks.plot_results import plot_tradeoff
    data = run_benchmark()
    plot_tradeoff(data, "results/tradeoff_curve.png")

def cmd_demo():
    from demo.search_demo import main
    main()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "benchmark":
        cmd_benchmark()
    elif mode == "demo":
        cmd_demo()
    elif mode == "all":
        cmd_benchmark()
        cmd_demo()
    else:
        print(f"Unknown mode: {mode}. Use: benchmark | demo | all")
