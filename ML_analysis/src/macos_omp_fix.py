"""
XGBoost's macOS wheel needs libomp.dylib, which normally requires Homebrew.
scikit-learn already ships its own bundled copy, so we point XGBoost at that
instead -- no Homebrew/sudo needed. macOS's dyld only reads DYLD_LIBRARY_PATH
at process *launch*, so setting os.environ mid-script has no effect; this
re-execs the current script once with the variable set before it, then
continues normally (call this before importing xgboost).
"""
import os
import sys


def ensure_xgboost_can_load():
    if sys.platform != "darwin" or os.environ.get("_AGSBTE2_OMP_FIXED"):
        return
    try:
        import sklearn
        dylibs = os.path.join(os.path.dirname(sklearn.__file__), ".dylibs")
    except ImportError:
        return
    if not os.path.isdir(dylibs):
        return
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = dylibs + ":" + env.get("DYLD_LIBRARY_PATH", "")
    env["_AGSBTE2_OMP_FIXED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
