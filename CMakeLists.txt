cmake_minimum_required(VERSION 3.16)
project(ParticleSimBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(main src/main.cpp)
