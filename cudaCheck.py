# cuda_check.py
import torch

def check_cuda():
    print("Checking CUDA availability...")

    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Memory Cached: {torch.cuda.memory_reserved()} bytes")
    else:
        print("❌ CUDA is NOT available.")
        print("Make sure you have a CUDA-compatible GPU and PyTorch is installed with CUDA support.")

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print("CUDA Available:", torch.cuda.is_available())

    check_cuda()
