"""Install the optional tools used by the T-count optimizer comparison.

This installer is intentionally opt-in.  The artifact's default
``--source existing`` mode never changes the Python environment or downloads
external repositories.  Use this module directly, or pass
``--install-missing`` to ``t_count_methods_comparison`` when generating new
results.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


TZAP_REPOSITORY = "https://github.com/qqq-wisc/tzap"
TZAP_CARGO_PACKAGE = "tzap-opt"
T_OPTIMIZER_REPOSITORY = "https://github.com/iqubit-org/T-Optimizer"
QUAEC_REPOSITORY = "https://github.com/cgranade/python-quaec"


def _default_external_root() -> Path:
    """Choose a writable artifact-local checkout directory."""

    working_tree_root = Path.cwd() / "micro_artifact"
    if working_tree_root.is_dir():
        return working_tree_root / ".external"
    return Path(__file__).resolve().parent / ".external"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not find required installer command {command[0]!r}."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Installer command failed with exit code {error.returncode}: "
            + " ".join(command)
        ) from error


def _command_exists(command: str) -> bool:
    parts = shlex.split(command)
    if not parts:
        return False
    return shutil.which(parts[0]) is not None or Path(parts[0]).expanduser().is_file()


def _install_python_packages(packages: Iterable[str]) -> None:
    selected = list(dict.fromkeys(packages))
    if selected:
        _run([sys.executable, "-m", "pip", "install", *selected])


def ensure_pyzx() -> str:
    """Install PyZX only when it cannot be imported."""

    was_available = importlib.util.find_spec("pyzx") is not None
    if not was_available:
        _install_python_packages(("pyzx",))
    if importlib.util.find_spec("pyzx") is None:
        raise RuntimeError("PyZX is still unavailable after installation.")
    return "available" if was_available else "installed"


def ensure_tzap(command: str | None = None) -> str:
    """Return a usable T-Zap command, installing its Cargo binary if needed."""

    configured = command or os.environ.get("TZAP_BIN", "tzap")
    if _command_exists(configured):
        return configured
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "T-Zap is missing and Cargo is unavailable. Install Rust/Cargo, then "
            f"run `cargo install {TZAP_CARGO_PACKAGE}`."
        )
    _run(["cargo", "install", TZAP_CARGO_PACKAGE])
    installed = shutil.which("tzap")
    if installed is None:
        raise RuntimeError(
            "T-Zap installation completed but the `tzap` executable is not on PATH. "
            "Add Cargo's bin directory to PATH or pass --tzap-bin."
        )
    return installed


def _ensure_checkout(repository: str, destination: Path, package_name: str) -> Path:
    if destination.is_dir() and (destination / package_name).is_dir():
        return destination
    if destination.exists():
        raise RuntimeError(
            f"Cannot install {package_name}: {destination} exists but is not a "
            "valid checkout. Choose another path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repository, str(destination)])
    if not (destination / package_name).is_dir():
        raise RuntimeError(
            f"Checkout at {destination} does not contain the expected {package_name}/ directory."
        )
    return destination


def ensure_t_optimizer(root: str | Path | None = None) -> Path:
    """Install T-Optimizer, QuaEC, and its Python build dependencies."""

    external_root = Path(root).expanduser() if root else _default_external_root() / "T-Optimizer"
    optimizer_root = _ensure_checkout(T_OPTIMIZER_REPOSITORY, external_root, "optimize")

    quaec_root = optimizer_root.parent / "QuaEC"
    # QuaEC's import package is named ``qecc`` even though the project is
    # distributed as QuaEC.
    if importlib.util.find_spec("qecc") is None:
        _ensure_checkout(QUAEC_REPOSITORY, quaec_root, "src")
        _run([sys.executable, "-m", "pip", "install", str(quaec_root)])

    missing = [
        package
        for module, package in (("numpy", "numpy"), ("gmpy2", "gmpy2"), ("Cython", "Cython"))
        if importlib.util.find_spec(module) is None
    ]
    _install_python_packages(missing)
    if not (optimizer_root / "optimize" / "T_optimizer.py").is_file():
        raise RuntimeError(
            f"T-Optimizer checkout at {optimizer_root} is missing optimize/T_optimizer.py."
        )
    return optimizer_root


def ensure_optimizers(
    methods: Iterable[str],
    *,
    tzap_bin: str | None = None,
    t_optimizer_root: str | Path | None = None,
) -> dict[str, str]:
    """Ensure selected methods are available and return their configuration."""

    selected = tuple(dict.fromkeys(methods))
    result: dict[str, str] = {}
    if "pyzx" in selected:
        ensure_pyzx()
        result["pyzx"] = "available"
    if "tzap" in selected:
        result["tzap_bin"] = ensure_tzap(tzap_bin)
    if "t-optimizer" in selected:
        result["t_optimizer_root"] = str(ensure_t_optimizer(t_optimizer_root))
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        choices=("pyzx", "tzap", "t-optimizer"),
        dest="methods",
        help="tool to install; repeat for multiple tools (default: all three)",
    )
    parser.add_argument("--tzap-bin", default=None, help="existing T-Zap executable or command")
    parser.add_argument(
        "--t-optimizer-root",
        type=Path,
        default=None,
        help="T-Optimizer checkout path (default: micro_artifact/.external/T-Optimizer)",
    )
    args = parser.parse_args()
    methods = args.methods or ["pyzx", "tzap", "t-optimizer"]
    try:
        result = ensure_optimizers(
            methods,
            tzap_bin=args.tzap_bin,
            t_optimizer_root=args.t_optimizer_root,
        )
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print("Optimizer setup complete.")
    for key, value in result.items():
        print(f"{key}={value}")
    if "t_optimizer_root" in result:
        print(f"Use --t-optimizer-root {result['t_optimizer_root']} for generated runs.")


if __name__ == "__main__":
    _main()
