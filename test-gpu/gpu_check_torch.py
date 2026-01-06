# file: gpu_check_torch.py
import time, torch, sys
def main_torch():
    print(f"PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if not cuda_ok:
        print("=> GPU 미인식. 드라이버/런타임/컨테이너 설정 확인 바람.")
        sys.exit(1)

    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[{i}] {torch.cuda.get_device_name(i)} | CC(compute capability) N/A")

    # 연산 테스트(행렬곱)
    N = 4096
    a_cpu = torch.randn((N, N))
    b_cpu = torch.randn((N, N))

    t0 = time.time()
    c_cpu = a_cpu @ b_cpu
    cpu_t = time.time() - t0

    device = torch.device("cuda:0")
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    torch.cuda.synchronize()
    t1 = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_t = time.time() - t1

    # 메모리/요약
    mem = torch.cuda.mem_get_info(device)
    print(f"GPU mem free/total (bytes): {mem[0]}/{mem[1]}")
    speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
    print(f"CPU: {cpu_t:.3f}s | GPU: {gpu_t:.3f}s | Speedup: x{speedup:.1f}")
    print("결론: PyTorch에서 GPU 연산 정상 동작.")
if __name__ == "__main__":
    main_torch()
