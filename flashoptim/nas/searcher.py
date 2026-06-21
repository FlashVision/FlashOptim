"""NAS search algorithms: random search and evolutionary search."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from flashoptim.nas.search_space import SearchSpace


class Searcher:
    """Neural Architecture Search engine.

    Supports random search and evolutionary (genetic algorithm) search
    strategies to find optimal architectures within a search space.

    Args:
        search_space: The architecture search space to explore.
        strategy: Search strategy ('random' or 'evolutionary').
        max_evals: Maximum number of architecture evaluations.
        population_size: Population size for evolutionary search.
        generations: Number of generations for evolutionary search.
        mutation_prob: Per-gene mutation probability.
        crossover_prob: Probability of crossover vs mutation.
    """

    STRATEGIES = ("random", "evolutionary")

    def __init__(
        self,
        search_space: SearchSpace,
        strategy: str = "evolutionary",
        max_evals: int = 100,
        population_size: int = 50,
        generations: int = 30,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.5,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Options: {self.STRATEGIES}")

        self.search_space = search_space
        self.strategy = strategy
        self.max_evals = max_evals
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        self._history: List[Dict[str, Any]] = []
        self._best: Optional[Dict[str, Any]] = None
        self._best_score: float = float("-inf")

    def search(self, evaluator: Any) -> Dict[str, Any]:
        """Run the NAS search loop.

        Args:
            evaluator: An :class:`Evaluator` instance that scores architectures.
                       Must implement ``evaluate(arch) -> dict`` with a 'score' key.

        Returns:
            Best architecture found during search.
        """
        if self.strategy == "random":
            return self._random_search(evaluator)
        return self._evolutionary_search(evaluator)

    def _random_search(self, evaluator: Any) -> Dict[str, Any]:
        """Random search: sample and evaluate independent architectures."""
        for i in range(self.max_evals):
            arch = self.search_space.sample()
            result = evaluator.evaluate(arch)
            score = result.get("score", result.get("accuracy", 0.0))

            self._history.append({"arch": arch, "result": result, "eval": i})

            if score > self._best_score:
                self._best_score = score
                self._best = arch

        return self._best

    def _evolutionary_search(self, evaluator: Any) -> Dict[str, Any]:
        """Evolutionary search with tournament selection, crossover, and mutation."""
        population = [self.search_space.sample() for _ in range(self.population_size)]
        fitness = []

        for arch in population:
            result = evaluator.evaluate(arch)
            score = result.get("score", result.get("accuracy", 0.0))
            fitness.append(score)
            self._history.append({"arch": arch, "result": result, "generation": 0})

            if score > self._best_score:
                self._best_score = score
                self._best = arch

        for gen in range(1, self.generations):
            new_population = []

            for _ in range(self.population_size):
                if random.random() < self.crossover_prob:
                    p1 = self._tournament_select(population, fitness)
                    p2 = self._tournament_select(population, fitness)
                    child = self.search_space.crossover(p1, p2)
                else:
                    parent = self._tournament_select(population, fitness)
                    child = self.search_space.mutate(parent, self.mutation_prob)

                new_population.append(child)

            population = new_population
            fitness = []
            for arch in population:
                result = evaluator.evaluate(arch)
                score = result.get("score", result.get("accuracy", 0.0))
                fitness.append(score)
                self._history.append({"arch": arch, "result": result, "generation": gen})

                if score > self._best_score:
                    self._best_score = score
                    self._best = arch

        return self._best

    @staticmethod
    def _tournament_select(
        population: List[Dict[str, Any]],
        fitness: List[float],
        k: int = 3,
    ) -> Dict[str, Any]:
        """Tournament selection: pick the best of k random candidates."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return the best architecture found so far.

        Returns:
            Best architecture dict, or None if search hasn't run.
        """
        return self._best

    @property
    def best_score(self) -> float:
        """Best score achieved during search."""
        return self._best_score

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full evaluation history."""
        return self._history

    def __repr__(self) -> str:
        return (
            f"Searcher(strategy={self.strategy}, max_evals={self.max_evals}, "
            f"pop_size={self.population_size}, gens={self.generations})"
        )
