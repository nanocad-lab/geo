cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sc)

find_package(Torch REQUIRED)

add_executable(sc sc.cpp)
target_link_libraries(sc "${TORCH_LIBRARIES}")
set_property(TARGET sc PROPERTY CXX_STANDARD 11)
