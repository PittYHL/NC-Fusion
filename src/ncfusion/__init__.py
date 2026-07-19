"""Public API and reproducibility harness for NC-Fusion."""

__version__ = "0.1.0"


def NC_Fusion(*args, **kwargs):
    """Compile a Hamiltonian and return its RZ and Clifford+T circuits."""
    from .api import NC_Fusion as _NC_Fusion

    return _NC_Fusion(*args, **kwargs)


__all__ = ["NC_Fusion", "__version__"]
