cmake_minimum_required(VERSION 3.31)
project(rayTracer)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(rayTracer
        src/rayTrace.h
        src/raytracer.cpp)
