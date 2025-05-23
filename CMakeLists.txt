cmake_minimum_required(VERSION 3.0...3.26)

project(hnsw-mips
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnsw-mips INTERFACE)
add_library(hnsw-mips::hnsw-mips ALIAS hnsw-mips)

target_include_directories(hnsw-mips INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnsw-mips
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnsw-mips
    EXPORT hnsw-mipsTargets)

install(EXPORT hnsw-mipsTargets
    FILE hnsw-mipsConfig.cmake
    NAMESPACE hnsw-mips::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnsw-mips)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnsw-mips)

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnsw-mips)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnsw-mips)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnsw-mips)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnsw-mips)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnsw-mips)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnsw-mips)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnsw-mips)

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnsw-mips)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnsw-mips)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnsw-mips)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnsw-mips)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnsw-mips)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnsw-mips)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnsw-mips)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnsw-mips)
endif()
