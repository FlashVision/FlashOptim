"""Search space definition for Neural Architecture Search."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


class SearchSpace:
    """Defines the set of searchable operations and architecture configurations.

    Encodes the design space for NAS, including channel widths, kernel sizes,
    network depths, and candidate operations at each decision point.

    Args:
        channels: Candidate channel widths (e.g. [16, 32, 64, 128, 256]).
        kernel_sizes: Candidate kernel sizes (e.g. [3, 5, 7]).
        depths: Candidate block depths per stage (e.g. [1, 2, 3, 4]).
        operations: Candidate operation types at each node.
    """

    BUILTIN_OPS = ("conv", "dwconv", "mbconv", "skip", "maxpool", "avgpool")

    def __init__(
        self,
        channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        operations: Optional[List[str]] = None,
    ) -> None:
        self.channels = channels or [16, 32, 64, 128, 256]
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.depths = depths or [1, 2, 3, 4]
        self.operations = operations or list(self.BUILTIN_OPS[:4])

    @property
    def num_stages(self) -> int:
        """Number of architecture stages (derived from channel options)."""
        return len(self.channels)

    def sample(self) -> Dict[str, Any]:
        """Sample a random architecture from the search space.

        Returns:
            Dictionary encoding a candidate architecture with keys:
            'channels', 'kernel_sizes', 'depths', 'operations'.
        """
        num_stages = random.randint(2, self.num_stages)
        arch = {
            "channels": [random.choice(self.channels) for _ in range(num_stages)],
            "kernel_sizes": [random.choice(self.kernel_sizes) for _ in range(num_stages)],
            "depths": [random.choice(self.depths) for _ in range(num_stages)],
            "operations": [random.choice(self.operations) for _ in range(num_stages)],
        }
        return arch

    def encode(self, arch: Dict[str, Any]) -> List[int]:
        """Encode an architecture dictionary to an integer vector.

        Each stage is encoded as indices into the respective candidate lists.

        Args:
            arch: Architecture dictionary from :meth:`sample`.

        Returns:
            Flat list of integer indices.
        """
        encoding = []
        num_stages = len(arch["channels"])
        for i in range(num_stages):
            encoding.append(self.channels.index(arch["channels"][i]))
            encoding.append(self.kernel_sizes.index(arch["kernel_sizes"][i]))
            encoding.append(self.depths.index(arch["depths"][i]))
            encoding.append(self.operations.index(arch["operations"][i]))
        return encoding

    def decode(self, encoding: List[int]) -> Dict[str, Any]:
        """Decode an integer vector back to an architecture dictionary.

        Args:
            encoding: Flat list of integer indices (groups of 4 per stage).

        Returns:
            Architecture dictionary.

        Raises:
            ValueError: If encoding length is not divisible by 4.
        """
        if len(encoding) % 4 != 0:
            raise ValueError(
                f"Encoding length must be divisible by 4, got {len(encoding)}"
            )

        arch: Dict[str, Any] = {"channels": [], "kernel_sizes": [], "depths": [], "operations": []}
        for i in range(0, len(encoding), 4):
            arch["channels"].append(self.channels[encoding[i]])
            arch["kernel_sizes"].append(self.kernel_sizes[encoding[i + 1]])
            arch["depths"].append(self.depths[encoding[i + 2]])
            arch["operations"].append(self.operations[encoding[i + 3]])
        return arch

    def mutate(self, arch: Dict[str, Any], prob: float = 0.1) -> Dict[str, Any]:
        """Mutate an architecture by randomly changing individual genes.

        Args:
            arch: Architecture to mutate.
            prob: Per-gene mutation probability.

        Returns:
            Mutated architecture (new dict, original unchanged).
        """
        mutated = {k: list(v) for k, v in arch.items()}
        num_stages = len(mutated["channels"])

        for i in range(num_stages):
            if random.random() < prob:
                mutated["channels"][i] = random.choice(self.channels)
            if random.random() < prob:
                mutated["kernel_sizes"][i] = random.choice(self.kernel_sizes)
            if random.random() < prob:
                mutated["depths"][i] = random.choice(self.depths)
            if random.random() < prob:
                mutated["operations"][i] = random.choice(self.operations)

        return mutated

    def crossover(
        self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single-point crossover between two parent architectures.

        Args:
            parent_a: First parent architecture.
            parent_b: Second parent architecture.

        Returns:
            Child architecture combining both parents.
        """
        min_stages = min(len(parent_a["channels"]), len(parent_b["channels"]))
        point = random.randint(1, max(min_stages - 1, 1))

        child: Dict[str, Any] = {}
        for key in ("channels", "kernel_sizes", "depths", "operations"):
            child[key] = list(parent_a[key][:point]) + list(parent_b[key][point:])
        return child

    def __repr__(self) -> str:
        return (
            f"SearchSpace(channels={self.channels}, kernels={self.kernel_sizes}, "
            f"depths={self.depths}, ops={self.operations})"
        )
