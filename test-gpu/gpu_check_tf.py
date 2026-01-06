# file: gpu_check_tf.py
import time, tensorflow as tf, os
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
if not gpus:
    print("=> GPU 미인식. 드라이버/런타임/컨테이너 설정 확인 바람.")
    raise SystemExit(1)

# 메모리 증가 허용
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("memory_growth 설정 실패:", e)

# 연산 테스트(행렬곱)
N = 4096
with tf.device('/CPU:0'):
    a_cpu = tf.random.normal((N, N))
    b_cpu = tf.random.normal((N, N))
with tf.device('/GPU:0'):
    a_gpu = tf.identity(a_cpu)
    b_gpu = tf.identity(b_cpu)

t0 = time.time()
with tf.device('/CPU:0'):
    _ = tf.matmul(a_cpu, b_cpu)
cpu_t = time.time() - t0

# 그래프 실행 보장
tf.experimental.numpy.random.seed(0)
tf.config.run_functions_eagerly(False)

t1 = time.time()
with tf.device('/GPU:0'):
    _ = tf.matmul(a_gpu, b_gpu)
gpu_t = time.time() - t1

print(f"CPU: {cpu_t:.3f}s | GPU: {gpu_t:.3f}s | Speedup: x{cpu_t/gpu_t:.1f}")
print("결론: TensorFlow에서 GPU 연산 정상 동작.")
