cmake_minimum_required(VERSION 3.10
    FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32 -mabi=ms -O0")
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -m32")

set(LUA_ROOT ${CMAKE_SOURCE_DIR}/../../deps/lua)

find_library(
    LUA_LIBS
    NAME liblua.so
    PATHS ${LUA_ROOT}
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_DEFAULT_PATH)

if (LUA_LIBS STREQUAL "LUA_LIBS-NOTFOUND")
    message(FAIL "Can't find lua at ${LUA_ROOT}")
else()
    message(STATUS "lua at ${LUA_LIBS}")
    set(LUA_INCLUDE_DIRS ${LUA_ROOT}/../..)
endif()


file(GLOB SRCS
    steam_api.cpp
    environment.cpp)

add_library(patch_toribash SHARED
    ${SRCS})

target_link_libraries(patch_toribash
    PRIVATE ${LUA_LIBS})

target_include_directories(patch_toribash
    PRIVATE ${LUA_INCLUDE_DIRS})

add_library(dummy SHARED
    dummy_so.cpp)

target_link_libraries(dummy
    PRIVATE ${LUA_LIBS})

target_include_directories(dummy
    PRIVATE ${LUA_INCLUDE_DIRS})
