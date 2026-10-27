cmake_minimum_required(VERSION 3.20)
project(quarry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # 只给 CXX：裸的 /W4 /permissive- 会被 nvcc 当成自己的参数，直接报错。
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
endif()

add_library(quarry_core STATIC
    src/core/scene.cpp
    src/cpu/brute.cpp
    src/cpu/bvh.cpp
    src/cpu/selftest.cpp
)
target_include_directories(quarry_core PUBLIC src)

add_executable(quarry src/cli/main.cpp)
target_link_libraries(quarry PRIVATE quarry_core)

add_executable(test_brute tests/test_brute.cpp)
target_link_libraries(test_brute PRIVATE quarry_core)

enable_testing()
add_test(NAME selftest COMMAND test_brute)

# 教程配套单文件：只保证能编译（CI 会构建全部 target），不参与 CTest 性能测量。
foreach(lesson 01_bruteforce 02_bvh 03_lbvh)
    add_executable(learn_${lesson} learn/${lesson}.cpp)
endforeach()

# GPU 后端。没有 CUDA 工具链的机器（CI、纯 CPU 环境）跳过整个块，
# CPU 参考实现和教程代码不受影响。
option(QUARRY_GPU "Build the CUDA backend when a toolkit is available" ON)
set(QUARRY_CUDA_ARCH "89" CACHE STRING "CUDA target architecture (e.g. 89, 86, 75)")

if(QUARRY_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        add_library(quarry_gpu STATIC
            src/gpu/lbvh.cu
            src/gpu/raycast.cu
        )
        target_include_directories(quarry_gpu PUBLIC src)
        target_link_libraries(quarry_gpu PUBLIC quarry_core PRIVATE CUDA::cudart)
        set_target_properties(quarry_gpu PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES ${QUARRY_CUDA_ARCH}
        )
        # 关掉乘加融合：CPU 侧是纯 IEEE 乘加，两个后端的 t 值要能逐位对拍。
        # 源码里有中文注释，主机编译器需要 /utf-8。
        target_compile_options(quarry_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --expt-relaxed-constexpr -Xcompiler=/utf-8>
        )

        add_executable(gpu_bench bench/gpu_bench.cpp)
        target_link_libraries(gpu_bench PRIVATE quarry_gpu)
        message(STATUS "CUDA ${CUDAToolkit_VERSION} found: building quarry_gpu (sm_${QUARRY_CUDA_ARCH})")
    else()
        message(STATUS "CUDA not found: building CPU-only")
    endif()
endif()
