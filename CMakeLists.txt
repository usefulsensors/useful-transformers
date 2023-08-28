cmake_minimum_required(VERSION 3.22)
project(CTransformer)

set(CMAKE_CXX_STANDARD 20)

option(NEON_OPT "enable neon optimizations" ON)
if (NEON_OPT)
  add_compile_definitions(NEON_OPT=true)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fPIC -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fPIC -march=native")

enable_testing()

add_subdirectory(lib)
add_subdirectory(examples/whisper)
