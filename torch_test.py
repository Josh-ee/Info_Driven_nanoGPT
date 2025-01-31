import torch
import time

def benchmark_pytorch():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        print(f"CUDA is available! Running on {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")

    # Define matrix size
    matrix_size = 10000  # Adjust size based on your GPU memory

    # CPU Benchmark
    a_cpu = torch.randn(matrix_size, matrix_size)
    b_cpu = torch.randn(matrix_size, matrix_size)
    
    print("\nBenchmarking on CPU...")
    start_time = time.time()
    result_cpu = torch.mm(a_cpu, b_cpu)  # Matrix multiplication on CPU
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.5f} seconds")

    if torch.cuda.is_available():
        # Move tensors to GPU
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # Warm-up GPU to avoid cold start effects
        for _ in range(5):
            _ = torch.mm(a_gpu, b_gpu)

        # GPU Benchmark
        print("\nBenchmarking on GPU...")
        torch.cuda.synchronize()  # Ensure previous operations are done
        start_time = time.time()
        result_gpu = torch.mm(a_gpu, b_gpu)  # Matrix multiplication on GPU
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.5f} seconds")

        print(f"\nSpeedup Factor (CPU/GPU): {cpu_time / gpu_time:.2f}x")

if __name__ == "__main__":
    benchmark_pytorch()
