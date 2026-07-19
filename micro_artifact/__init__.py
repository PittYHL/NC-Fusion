"""MICRO-style evaluation entry points for NC-Fusion.

Each paper evaluation has its own module in this package.  The modules keep
their command-line interfaces small and delegate common output handling to
``micro_artifact.common``.
"""

from .common import EVALUATIONS, MissingEvaluationError

__all__ = ["EVALUATIONS", "MissingEvaluationError"]
