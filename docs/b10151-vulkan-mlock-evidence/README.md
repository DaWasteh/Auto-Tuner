# b10151 Vulkan GPU-locking evidence

- Binary: `L:\tmp\llama-b10151-release\bin\llama-server.exe` (`llama-server --version` → `10151 (8e8681e0e)`)
- Backend: Clang 20.1.8 Windows Vulkan x86_64
- Model: `fastcontext-1.0-4b-rl-q8_0.gguf` (3.99 GiB), `-ngl 999` (full GPU offload)
- System: Intel(R) Core(TM) Ultra 9 285K, 2 GPU(s)
  - AMD Radeon AI PRO R9700 (32624 MB VRAM)
  - AMD Radeon RX 9070 XT (16304 MB VRAM)

- `--load-mode mlock`: ✅ loaded + completion OK — see `b10151-mlock.log`
- `--load-mode mmap+mlock`: ✅ loaded + completion OK — see `b10151-mmap-mlock.log`
