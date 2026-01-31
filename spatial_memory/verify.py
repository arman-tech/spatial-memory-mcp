"""Verify spatial-memory installation and dependencies.

Run: python -m spatial_memory.verify
"""

import contextlib
import io
import logging
import os
import sys
import warnings


def _suppress_noise() -> None:
    """Suppress noisy warnings and logs from dependencies."""
    warnings.filterwarnings("ignore")
    # Suppress verbose loggers
    for logger_name in [
        "sentence_transformers",
        "optimum",
        "huggingface_hub",
        "transformers",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    # Suppress HF symlink warnings on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main() -> int:
    """Verify installation and print status."""
    _suppress_noise()

    print("Spatial Memory MCP - Installation Verification")
    print("=" * 50)

    errors = []
    onnx_available = False

    # Check core dependencies
    print("\nChecking core dependencies...")

    try:
        import lancedb  # noqa: F401
        print("  [OK] lancedb")
    except ImportError as e:
        print(f"  [FAIL] lancedb: {e}")
        errors.append("lancedb")

    try:
        import sentence_transformers  # noqa: F401
        print("  [OK] sentence-transformers")
    except ImportError as e:
        print(f"  [FAIL] sentence-transformers: {e}")
        errors.append("sentence-transformers")

    try:
        import mcp  # noqa: F401
        print("  [OK] mcp")
    except ImportError as e:
        print(f"  [FAIL] mcp: {e}")
        errors.append("mcp")

    try:
        import hdbscan  # noqa: F401
        print("  [OK] hdbscan")
    except ImportError as e:
        print(f"  [FAIL] hdbscan: {e}")
        errors.append("hdbscan")

    try:
        import umap  # noqa: F401
        print("  [OK] umap-learn")
    except ImportError as e:
        print(f"  [FAIL] umap-learn: {e}")
        errors.append("umap-learn")

    # Check ONNX Runtime (optional but recommended)
    print("\nChecking ONNX Runtime (for faster embeddings)...")

    try:
        import onnxruntime  # noqa: F401
        print("  [OK] onnxruntime")
    except ImportError:
        print("  [WARN] onnxruntime not installed")

    # Suppress "Multiple distributions" message from optimum
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            import optimum.onnxruntime  # noqa: F401
            optimum_ok = True
        except ImportError:
            optimum_ok = False

    if optimum_ok:
        print("  [OK] optimum")
    else:
        print("  [WARN] optimum not installed")

    # Check embedding service
    print("\nChecking embedding service...")

    try:
        from spatial_memory.core.embeddings import EmbeddingService, _is_onnx_available

        onnx_available = _is_onnx_available()
        print(f"  ONNX Runtime available: {onnx_available}")

        # Suppress model loading messages
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            svc = EmbeddingService()
            backend = svc.backend
            dimensions = svc.dimensions
            vec = svc.embed("test")

        print(f"  Active backend: {backend}")
        print(f"  Embedding dimensions: {dimensions}")
        print(f"  Embedding test: OK (shape: {vec.shape})")

    except Exception as e:
        print(f"  [FAIL] Embedding service error: {e}")
        errors.append("embedding-service")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} issue(s) found")
        print(f"Missing: {', '.join(errors)}")
        print("\nTry: pip install -e \".[dev]\"")
        return 1
    else:
        print("SUCCESS: All dependencies verified!")
        if not onnx_available:
            print("\nTip: Install ONNX for 2-3x faster embeddings:")
            print("  pip install sentence-transformers[onnx]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
