import torch

def check_environment():
    """Перевірка середовища виконання"""
    print("=" * 60)
    print("ІНФОРМАЦІЯ ПРО СЕРЕДОВИЩЕ")
    print("=" * 60)

    # PyTorch версія
    print(f"📦 PyTorch версія: {torch.__version__}")

    # CUDA доступність
    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA доступна: {cuda_available}")

    if cuda_available:
        print(f"🔧 CUDA версія: {torch.version.cuda}")
        print(f"📊 Кількість GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Назва: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Пам'ять: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("⚠️  GPU не знайдено. Використовуємо CPU.")
        print("Для активації GPU в Colab: Runtime -> Change runtime type -> GPU")

    # CPU інформація
    print(f"\n💻 CPU threads: {torch.get_num_threads()}")

    return cuda_available


# Запуск перевірки
cuda_available = check_environment()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"\n✅ Використовуємо device: {device}\n")
