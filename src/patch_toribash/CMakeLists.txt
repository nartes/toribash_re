cmake_minimum_required(VERSION 3.10
	FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32 -mabi=ms -O0")
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -m32")

file(GLOB SRCS
	*.cpp)

add_library(patch_toribash SHARED
	${SRCS})

add_library(dummy SHARED
	dummy_so.cpp)
