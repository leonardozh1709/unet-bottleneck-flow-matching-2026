from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"     # trained models + their training logs
DATA_DIR = REPO_ROOT / "data"           # CIFAR-10
PLOTS_DIR = REPO_ROOT / "plots"         # every metric plot, one subfolder per metric
CACHE_DIR = PLOTS_DIR / "cache"         # cached metric results (see Utils/seeds.py)
