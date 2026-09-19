# cayleypy is used ONLY for the beam-search benchmark / solve cells at the end
# (training below is self-contained PyTorch). Same pinned commit as the other
# community baseline notebooks.
import importlib.util

if importlib.util.find_spec('cayleypy') is None:
    get_ipython().system('pip install git+https://github.com/cayleypy/cayleypy.git@e1518b6 --no-deps')
else:
    print('cayleypy already installed')
