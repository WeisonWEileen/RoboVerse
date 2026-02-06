import sys
import platform

def main():
    print("=== PyTorch sanity check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except Exception as e:
        print("\n[FAIL] Could not import torch.")
        print("Reason:", repr(e))
        sys.exit(1)

    print("\n[OK] Imported torch")
    print("torch.__version__:", getattr(torch, "__version__", "unknown"))

    # Basic CPU test
    try:
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        c = a @ b  # matmul
        s = c.mean().item()
        print("\n[OK] CPU tensor ops work")
        print("CPU matmul mean:", s)
    except Exception as e:
        print("\n[FAIL] CPU tensor ops failed.")
        print("Reason:", repr(e))
        sys.exit(2)

    # CUDA test (if available)
    try:
        cuda_available = torch.cuda.is_available()
        print("\nCUDA available:", cuda_available)

        if cuda_available:
            dev = torch.device("cuda:0")
            x = torch.randn(4096, 4096, device=dev)
            y = torch.randn(4096, 4096, device=dev)
            z = x @ y
            torch.cuda.synchronize()
            print("[OK] CUDA tensor ops work on:", torch.cuda.get_device_name(0))
            print("CUDA matmul mean:", z.mean().item())
        else:
            print("[INFO] Skipping CUDA test (not available).")
    except Exception as e:
        print("\n[FAIL] CUDA check/op failed.")
        print("Reason:", repr(e))
        sys.exit(3)

    print("\n=== PASS: PyTorch looks functional ===")
    sys.exit(0)

if __name__ == "__main__":
    main()