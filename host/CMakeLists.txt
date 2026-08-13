cmake_minimum_required(VERSION 3.16)
project(needle_host C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../engine)

file(GLOB ENGINE_SRC ${ENGINE_DIR}/src/*.c)

add_library(needle_engine STATIC ${ENGINE_SRC})
target_include_directories(needle_engine PUBLIC ${ENGINE_DIR}/include)
target_link_libraries(needle_engine PUBLIC m)

add_executable(nd_dump nd_dump.c)
target_link_libraries(nd_dump PRIVATE needle_engine)

add_executable(nd_gtest nd_gtest.c)
target_link_libraries(nd_gtest PRIVATE needle_engine)
