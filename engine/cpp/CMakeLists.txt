cmake_minimum_required(VERSION 3.18)
project(inference_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and PyTorch
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(SOURCES
    src/iteration_loop.cpp
    src/batch_builder.cpp
    src/bindings.cpp
)

# Build the Python extension
pybind11_add_module(inference_engine_cpp ${SOURCES})

target_include_directories(inference_engine_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(inference_engine_cpp PRIVATE
    ${TORCH_LIBRARIES}
)

# Install
install(TARGETS inference_engine_cpp DESTINATION .)
