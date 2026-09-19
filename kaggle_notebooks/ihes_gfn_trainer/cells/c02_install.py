# --- Setup: use Kaggle's PRE-INSTALLED jax/libtpu; add only pure-Python libs ---
# Do NOT pip-reinstall jax on a Kaggle TPU VM - it breaks the libtpu binding.
# Enable "TPU VM v3-8" + Internet in the notebook settings (right pane).
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "equinox", "optax"],
               check=False)

import jax
print("jax", jax.__version__, "devices:", jax.device_count(),
      jax.devices()[0].device_kind)
