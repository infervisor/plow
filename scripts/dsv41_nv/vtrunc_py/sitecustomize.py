# Truncated-model vLLM runs: drop checkpoint weights of layers >= PLOW_VTRUNC_LAYERS in
# DeepseekV4Model.load_weights (the stock loader raises KeyError on them).
import importlib.abc
import os
import re
import sys

_N = os.environ.get("PLOW_VTRUNC_LAYERS")
_MOD = "vllm.models.deepseek_v41.nvidia.model"
_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _patch(m):
    cls = m.DeepseekV4Model
    orig = cls.load_weights
    n = int(_N)

    def load_weights(self, weights):
        def keep(name):
            g = _RE.search(name)
            return g is None or int(g.group(1)) < n
        return orig(self, ((k, v) for k, v in weights if keep(k)))

    cls.load_weights = load_weights


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MOD:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        sys.meta_path.insert(0, self)
        if spec is None:
            return None
        ex = spec.loader.exec_module

        def exec_module(module):
            ex(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _N:
    import importlib.util
    sys.meta_path.insert(0, _Finder())
