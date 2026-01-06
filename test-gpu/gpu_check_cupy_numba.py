# file: gpu_check_cupy_numba.py
import time, sys
try:
    import cupy as cp
    print("CuPy version:", cp.__version__)
    dev = cp.cuda.Device()
    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    N = 4096
    a = cp.random.randn(N, N)
    b = cp.random.randn(N, N)
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    c = a @ b
    cp.cuda.Stream.null.synchronize()
    print(f"CuPy GEMM(GPU) time: {time.time()-t0:.3f}s")
    print("결론: CuPy로 GPU 연산 정상 동작.")
except Exception as e:
    print("CuPy 실패:", e)
    print("→ Numba로 커널 테스트 시도")

    from numba import cuda
    import numpy as np
    if not cuda.is_available():
        print("Numba: CUDA 미가용.")
        sys.exit(1)

    @cuda.jit
    def add_kernel(x, y, out):
        i = cuda.grid(1)
        if i < x.size:
            out[i] = x[i] + y[i]

    n = 10_000_000
    x = np.ones(n, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)
    d_x = cuda.to_device(x); d_y = cuda.to_device(y); d_o = cuda.device_array_like(x)
    threads = 256
    blocks = (n + threads - 1) // threads
    start = time.time()
    add_kernel[blocks, threads](d_x, d_y, d_o)
    cuda.synchronize()
    print(f"Numba 커널 실행 시간: {time.time()-start:.3f}s")
    print("결론: Numba CUDA 커널 정상 동작.")
