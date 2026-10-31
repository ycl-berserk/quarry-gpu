cmake_minimum_required(VERSION 3.20)
project(quarry_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 对安装前缀编译：
#   cmake -S examples -B examples/build -DCMAKE_PREFIX_PATH=C:\quarry
find_package(quarry 1.0 CONFIG REQUIRED)

add_executable(quarry_example main.cpp)
if(TARGET quarry::gpu)
    target_link_libraries(quarry_example PRIVATE quarry::gpu)
    target_compile_definitions(quarry_example PRIVATE QUARRY_EXAMPLE_GPU=1)
    message(STATUS "quarry example: GPU backend")
else()
    target_link_libraries(quarry_example PRIVATE quarry::core)
    message(STATUS "quarry example: CPU-only")
endif()
