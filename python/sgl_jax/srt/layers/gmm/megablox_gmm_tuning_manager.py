import json
import os
import threading
from typing import Dict, Optional

import numpy as np


def get_default_cache_dir():
    return os.environ.get("GMM_TUNING_CACHE_DIR", "/tmp/gmm_tuning_cache")


class MegabloxGMMTuningManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cache_dir: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if self._initialized:
            return

        self.cache_dir = cache_dir or get_default_cache_dir()
        self.tuning_cache: Dict[str, np.ndarray[np.int32]] = {}
        self.default_tuning_result = np.array([512, 1024, 1024], dtype=np.int32)
        self._load_all_cached_tuning_results()
        self._initialized = True

    def _load_all_cached_tuning_results(self):
        if not os.path.exists(self.cache_dir):
            return

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                cache_key = filename[:-5]  # Remove .json extension
                cache_file = os.path.join(self.cache_dir, filename)

                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        if "tuning_result" in data:
                            self.tuning_cache[cache_key] = tuple(data["tuning_result"])
                except Exception:
                    continue

    def get_tuning_result(
        self, m: int, k: int, n: int, num_groups: int
    ) -> np.ndarray[np.int32]:
        cache_key = self._get_cache_key(m, k, n, num_groups)

        if cache_key in self.tuning_cache:
            return self.tuning_cache[cache_key]

        for cached_key, tiling in self.tuning_cache.items():
            parts = cached_key.split("_")
            if len(parts) == 4:
                try:
                    cached_m = int(parts[0][1:])
                    cached_k = int(parts[1][1:])
                    cached_n = int(parts[2][1:])
                    cached_groups = int(parts[3][1:])

                    if (
                        cached_k == k
                        and cached_n == n
                        and cached_groups == num_groups
                        and (
                            abs(cached_m - m) / max(cached_m, m) < 0.5
                            or min(cached_m, m) <= 256
                        )
                    ):
                        return tiling
                except ValueError:
                    continue

        return self.default_tuning_result

    def _get_cache_key(self, m: int, k: int, n: int, num_groups: int) -> str:
        return f"m{m}_k{k}_n{n}_g{num_groups}"


_global_megablox_gmm_tuning_manager = MegabloxGMMTuningManager()


def get_global_megablox_gmm_tuning_manager(
    cache_dir: Optional[str] = None,
) -> MegabloxGMMTuningManager:
    global _global_megablox_gmm_tuning_manager
    if _global_megablox_gmm_tuning_manager is None:
        _global_megablox_gmm_tuning_manager = MegabloxGMMTuningManager(cache_dir)
    return _global_megablox_gmm_tuning_manager


def get_tuning_result(m: int, k: int, n: int, num_groups: int) -> np.ndarray[np.int32]:
    manager = get_global_megablox_gmm_tuning_manager()
    return manager.get_tuning_result(m, k, n, num_groups)
