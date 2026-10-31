import importlib
import sys

# Config files saved before the repo reorganisation (outputs/*/config_*.pkl) reference the
# old top-level module name "config". This alias keeps them loadable without touching them.
sys.modules.setdefault("config", importlib.import_module("Utils.config"))
