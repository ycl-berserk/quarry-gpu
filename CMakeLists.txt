cmake_minimum_required(VERSION 3.20)
project(quarry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
    add_compile_options($<$<CONFIG:Release>:/O2>)
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

# GPU 部分在 M0 接入，届时在这里加 LANGUAGES CUDA 和 src/gpu/*.cu。
