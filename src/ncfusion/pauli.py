"""Small deterministic Pauli helpers used by the artifact smoke test."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


VALID_PAULIS = frozenset("IXYZ")


def validate_pauli(label: str) -> str:
    if not label or any(char not in VALID_PAULIS for char in label):
        raise ValueError(f"Invalid Pauli string: {label!r}")
    return label


def commutes(first: str, second: str) -> bool:
    """Return whether two equal-width Pauli strings commute."""

    validate_pauli(first)
    validate_pauli(second)
    if len(first) != len(second):
        raise ValueError("Pauli strings must have equal width")
    anticommutes = sum(
        left != "I" and right != "I" and left != right
        for left, right in zip(first, second)
    )
    return anticommutes % 2 == 0


def support(label: str) -> tuple[int, ...]:
    validate_pauli(label)
    return tuple(index for index, char in enumerate(label) if char != "I")


def group_by_support(labels: Iterable[str]) -> dict[tuple[int, ...], list[str]]:
    """Group labels deterministically by their non-identity support."""

    grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for label in labels:
        grouped[support(label)].append(label)
    return dict(sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])))


def anticommuting_pairs(labels: Iterable[str]) -> int:
    values = list(labels)
    return sum(not commutes(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values)))

