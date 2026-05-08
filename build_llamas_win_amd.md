´´´ Main-Fork ´´´
cd C:\LAB\ai-local
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_ASM_COMPILER="cl.exe"
cmake --build build --config Release -j 24

´´´ PrismML-Fork ´´´
cd C:\LAB\ai-local
git clone https://github.com/PrismML-Eng/llama.cpp 1b_llama.cpp
cd 1b_llama.cpp
git checkout prism
cmake -B build -DGGML_VULKAN=ON -DCMAKE_ASM_COMPILER="cl.exe"
cmake --build build --config Release -j 24

´´´ Ikawrakow MTP-Fork ´´´
cd C:\LAB\ai-local
git clone https://github.com/ikawrakow/ik_llama.cpp ik_llama.cpp
cd ik_llama.cpp
git fetch origin
git checkout ik/gemma4
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_COMPILER="cl.exe"
cmake --build build --config Release -j 24

´´´ TurboQuant-Fork ´´´
cd C:\LAB\ai-local
git clone https://github.com/TheTom/llama-cpp-turboquant tq_llama.cpp
cd tq_llama.cpp
git checkout thetom
cmake -B build -DGGML_VULKAN=ON -DCMAKE_ASM_COMPILER="cl.exe"
cmake --build build --config Release -j 24

´´´ Atomic MTP-Fork ´´´
cd C:\LAB\ai-local
git clone https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant atq_llama.cpp
cd atq_llama.cpp
git checkout Atomic
cmake -B build -DGGML_VULKAN=ON -DCMAKE_ASM_COMPILER="cl.exe"
cmake --build build --config Release -j 24
