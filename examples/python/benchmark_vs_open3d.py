import time
import numpy as np

points = np.random.randn(1_000_000, 3).astype(np.float32)
start = time.perf_counter()
_ = points[::2]
elapsed_ms = (time.perf_counter() - start) * 1000
print(f"Placeholder benchmark: {elapsed_ms:.2f}ms")
