cmake_minimum_required(VERSION 3.10)
project(dcgan VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(dcgan 
    src/main.cpp 
    src/dcgan.cpp 
    src/train.cpp
)
target_compile_definitions(dcgan PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(dcgan PRIVATE headers)
target_include_directories(dcgan PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(dcgan "${TORCH_LIBRARIES}")
target_link_libraries(dcgan ${OpenCV_LIBS})

if(MSVC)
  add_compile_options(/W0)
else()
  add_compile_options(-w)
endif()