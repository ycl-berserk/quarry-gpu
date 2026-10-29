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
    src/core/point_cloud.cpp
    src/core/gaussian.cpp
    src/cpu/brute.cpp
    src/cpu/bvh.cpp
    src/cpu/point_kd.cpp
    src/cpu/gauss_bvh.cpp
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
            src/gpu/query.cu
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

        add_executable(gpu_query_bench bench/gpu_query_bench.cpp)
        target_link_libraries(gpu_query_bench PRIVATE quarry_gpu)

        add_executable(gpu_cloud_bench bench/gpu_cloud_bench.cpp)
        target_link_libraries(gpu_cloud_bench PRIVATE quarry_gpu)

        add_executable(gpu_gauss_bench bench/gpu_gauss_bench.cpp)
        target_link_libraries(gpu_gauss_bench PRIVATE quarry_gpu)

        # Python 绑定：pybind11 由 pip 安装时不是系统包，向解释器要 cmakedir。
        # 没装就跳过，CI（无 CUDA）和纯 C++ 构建都不受影响。
        option(QUARRY_PYTHON "Build the Python bindings when pybind11 is available" ON)
        if(QUARRY_PYTHON)
            find_package(Python COMPONENTS Interpreter Development.Module QUIET)
            if(Python_FOUND)
                execute_process(
                    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE QUARRY_PYBIND11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE QUARRY_PYBIND11_RC)
            endif()
            if(Python_FOUND AND QUARRY_PYBIND11_RC EQUAL 0)
                find_package(pybind11 CONFIG QUIET PATHS ${QUARRY_PYBIND11_DIR} NO_DEFAULT_PATH)
            endif()
            if(pybind11_FOUND)
                pybind11_add_module(quarry_py MODULE bindings/python/quarry_py.cpp)
                target_link_libraries(quarry_py PRIVATE quarry_gpu)
                set_target_properties(quarry_py PROPERTIES
                    OUTPUT_NAME quarry
                    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
                    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
                # Python 3.8+ 不再用 PATH 找扩展模块的依赖，把 cudart 放到 .pyd 旁边，
                # 免去在每个脚本里 os.add_dll_directory。
                file(GLOB QUARRY_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
                if(QUARRY_CUDART_DLLS)
                    add_custom_command(TARGET quarry_py POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                ${QUARRY_CUDART_DLLS} $<TARGET_FILE_DIR:quarry_py>)
                endif()
                message(STATUS "Python ${Python_VERSION}: building quarry bindings")
            else()
                message(STATUS "pybind11 not found: skipping Python bindings")
            endif()
        endif()

        message(STATUS "CUDA ${CUDAToolkit_VERSION} found: building quarry_gpu (sm_${QUARRY_CUDA_ARCH})")
    else()
        message(STATUS "CUDA not found: building CPU-only")
    endif()
endif()
