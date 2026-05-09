"""
Reproducibility utilities.

Set deterministic seeds and configure PyTorch / NumPy / Python random
in one place so every run produces the same numbers given the same
inputs and the same hardware.

Use:
    from reproducibility import set_global_seed
    set_global_seed(42)
"""
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """Set all relevant seeds and force deterministic algorithms.

    Notes:
    - Sets Python, NumPy, and PyTorch CPU/CUDA seeds.
    - Sets PYTHONHASHSEED so dict ordering is deterministic across runs.
    - Configures cuDNN to deterministic mode (slower but reproducible).
    - Sets `CUBLAS_WORKSPACE_CONFIG` so PyTorch's deterministic algorithms
      can be enabled without runtime errors.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms (raises if a non-deterministic op is used).
    # We use warn_only=True to avoid breaking on PyG ops that lack determ. impls.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch without warn_only kwarg
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def report_environment():
    """Print versions and hardware for the run log."""
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"NumPy:  {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        import torch_geometric
        print(f"PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        pass
    try:
        import mne
        print(f"MNE: {mne.__version__}")
    except ImportError:
        pass
