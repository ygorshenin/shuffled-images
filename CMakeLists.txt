cmake_minimum_required(VERSION 3.6)

project(Solver)

find_package(gflags REQUIRED)

option(USE_PPROF OFF)
option(USE_ASAN OFF)

add_compile_options(
  "-std=c++1z"
  "-Wall"
)

if (USE_ASAN)
  add_compile_options(
    "-fno-omit-frame-pointer"
    "-fsanitize=address"
  )
endif()

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} gflags)

if (USE_PPROF)
  target_link_libraries(${PROJECT_NAME} "-lprofiler")
endif()

if (USE_ASAN)
  target_link_libraries(${PROJECT_NAME} "-fsanitize=address")
endif()
