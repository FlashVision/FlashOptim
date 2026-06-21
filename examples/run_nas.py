#!/usr/bin/env python3
"""Example: Neural Architecture Search with FlashOptim.

Demonstrates how to define a search space, run an evolutionary NAS search,
and evaluate candidate architectures.

Usage:
    python examples/run_nas.py
"""

from flashoptim.nas import SearchSpace, Searcher, Evaluator


def main():
    print("=" * 60)
    print("FlashOptim — Neural Architecture Search Example")
    print("=" * 60)

    search_space = SearchSpace(
        channels=[16, 32, 64, 128],
        kernel_sizes=[3, 5],
        depths=[1, 2, 3],
        operations=["conv", "dwconv", "mbconv", "skip"],
    )
    print(f"\nSearch Space: {search_space}")

    print("\n--- Sampling Random Architectures ---")
    for i in range(3):
        arch = search_space.sample()
        encoding = search_space.encode(arch)
        decoded = search_space.decode(encoding)
        print(f"  Arch {i+1}: channels={arch['channels']}, ops={arch['operations']}")
        print(f"           encoding={encoding}")
        assert decoded == arch, "Encode/decode round-trip failed"

    print("\n--- Mutation & Crossover ---")
    parent_a = search_space.sample()
    parent_b = search_space.sample()
    mutated = search_space.mutate(parent_a, prob=0.5)
    child = search_space.crossover(parent_a, parent_b)
    print(f"  Parent A: {parent_a['channels']}")
    print(f"  Mutated:  {mutated['channels']}")
    print(f"  Child:    {child['channels']}")

    print("\n--- Evaluating Architectures (no training data) ---")
    evaluator = Evaluator(
        proxy_epochs=1,
        device="cpu",
        max_flops=1e10,
        max_params=5e6,
    )

    arch = search_space.sample()
    result = evaluator.evaluate(arch)
    print(f"  Architecture: channels={arch['channels']}")
    print(f"  Result: params={result['params']:,}, latency={result['latency_ms']:.2f}ms")
    print(f"  Feasible: {result['feasible']}, Score: {result['score']:.4f}")

    print("\n--- Running Random Search (5 evaluations) ---")
    searcher = Searcher(
        search_space=search_space,
        strategy="random",
        max_evals=5,
    )
    best = searcher.search(evaluator)
    print(f"  Best architecture: channels={best['channels']}")
    print(f"  Best score: {searcher.best_score:.4f}")
    print(f"  Total evaluations: {len(searcher.history)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
