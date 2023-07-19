import os
import uuid
import pickle
import functools
import inspect

def stringify_cache_key(key):
    return uuid.uuid3(uuid.NAMESPACE_DNS, str(key)).hex

def cache(cache_key, version=0.0, disable=False):
    """Cache the result of a function call on disk for speedup"""
    def inner_cache(f):
        f_sig = inspect.signature(f)
        @functools.wraps(f)
        def cached_f(cfg, *args, **kwargs):

            # ensure that default args are properly passed to cache key
            bound = f_sig.bind(cfg, *args, **kwargs)
            bound.apply_defaults()
            _, *args = bound.args
            kwargs = bound.kwargs

            key = stringify_cache_key(cache_key(cfg, *args, **kwargs))
            cache_file = f"{cfg.platform.cache_dir}/functions/{f.__name__}/{version}/{key}.pkl"
            if not disable:
                try:
                    with open(cache_file, "rb") as fh:
                        ret = pickle.load(fh)
                        return ret
                except (FileNotFoundError, EOFError):
                    pass
            ret = f(cfg, *args, **kwargs)
            cache_folder = "/".join(cache_file.split("/")[:-1])
            os.makedirs(cache_folder, exist_ok=True)
            with open(cache_file, "wb") as fh:
                pickle.dump(ret, fh)
                return ret
        return cached_f
    return inner_cache