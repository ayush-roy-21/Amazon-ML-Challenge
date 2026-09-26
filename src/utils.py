import time

_T0 = time.time()


def log(*args):
    print(f"[{time.time() - _T0:8.1f}s]", *args, flush=True)
