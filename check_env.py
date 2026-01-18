import torch

def check_environment():
    """Prints environment info. Returns True if CUDA is available."""
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)

    # PyTorch
    print(f"📦 PyTorch version: {torch.__version__}\n")

    # CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"🖥️  CUDA is available")
        print(f"🔧 CUDA version: {torch.version.cuda}")
        print(f"📊 GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU №{i + 1}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("⚠️  GPU is not available")

    # CPU
    print(f"\n💻 CPU threads: {torch.get_num_threads()}")

    return cuda_available


# main
if __name__=='__main__':
    cuda_available = check_environment()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"\n✅ Device: {device} is being used\n")
