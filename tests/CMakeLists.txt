cmake_minimum_required(VERSION 3.5.1)

project(UnitTests)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Sanitizers) # Address sanitizer.

if (NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/../third_party/Catch/include")
  message(FATAL_ERROR "RainbowMist dependency repositories (Catch, etc.) are missing! "
      "You probably did not clone the project with --recursive. It is possible to recover "
      "by calling \"git submodule update --init --recursive\"")
endif()


# Add custom build type DebugOpt
if (MSVC)
set(CMAKE_CXX_FLAGS_DEBUGOPT
    "-DDEBUG /DEBUG /O2"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )
set(CMAKE_C_FLAGS_DEBUGOPT
    "-DDEBUG /DEBUG /O2"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE )
else () # Assume gcc
set(CMAKE_CXX_FLAGS_DEBUGOPT
    "-g -O2 -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )
set(CMAKE_C_FLAGS_DEBUGOPT
    "-g -O2 -fno-omimt-frame-pointer"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE )
endif()

set(CMAKE_EXE_LINKER_FLAGS_DEBUGOPT
    ""
    CACHE STRING "Flags used for linking binaries during coverage builds."
    FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGOPT
    ""
    CACHE STRING "Flags used by the shared libraries linker during coverage builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_DEBUGOPT
    CMAKE_C_FLAGS_DEBUGOPT
    CMAKE_EXE_LINKER_FLAGS_DEBUGOPT
    CMAKE_SHARED_LINKER_FLAGS_DEBUGOPT )

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel DebugOpt."
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Lucia: Current build type is : ${CMAKE_BUILD_TYPE}")

# C++11
set (CMAKE_CXX_STANDARD 11)

# Sanitizer
find_package(Sanitizers)

# Threads
find_package(Threads)

# [glm]
include_directories("${CMAKE_SOURCE_DIR}/../third_party/glm/include")

# [CxxSwizzle]
include_directories("${CMAKE_SOURCE_DIR}/../third_party/CxxSwizzle/include")

# [CLCudaAPI]
include_directories("${CMAKE_SOURCE_DIR}/CLCudaAPI/include")

# [EasyCL]
set(USE_CLEW On)
add_definitions(-DUSE_CLEW)
include_directories("${CMAKE_SOURCE_DIR}/EasyCL")
include_directories("${CMAKE_SOURCE_DIR}/EasyCL/thirdparty/clew/include")
include_directories("${CMAKE_SOURCE_DIR}/EasyCL/thirdparty/lua-5.1.5/src")

file(STRINGS "${CMAKE_SOURCE_DIR}/EasyCL/thirdparty/lua-5.1.5/files.txt" lua_src1)
foreach(source ${lua_src1})
    set(lua_src ${lua_src} "${CMAKE_SOURCE_DIR}/EasyCL/thirdparty/lua-5.1.5/src/${source}")
endforeach()
add_library(EasyCL
    "${CMAKE_SOURCE_DIR}/EasyCL/CLKernel.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/CLWrapper.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/DeviceInfo.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/DevicesInfo.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/EasyCL.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/platforminfo_helper.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/deviceinfo_helper.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/util/easycl_stringhelper.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/util/StatefulTimer.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/templates/LuaTemplater.cpp"
    "${CMAKE_SOURCE_DIR}/EasyCL/templates/TemplatedKernel.cpp"
    ${lua_src}

)

# force static build
add_definitions(-Dclew_STATIC)
add_library(clew
    "${CMAKE_SOURCE_DIR}/EasyCL/thirdparty/clew/src/clew.c"
)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../
    ${CMAKE_SOURCE_DIR}/../third_party/glm/
    ${CMAKE_SOURCE_DIR}/cuew/include
    ${CMAKE_SOURCE_DIR}/EasyCL
    ${CMAKE_SOURCE_DIR}/../third_party/Catch/include/
)

add_definitions(-DUSE_CUEW)
include_directories("${CMAKE_SOURCE_DIR}/cuew/include")


if (MSVC)
  add_definitions( "-DNOMINMAX" )
endif ()

set (TEST_SOURCE
    ${CMAKE_SOURCE_DIR}/main.cc
    )

add_library(cuew
  "${CMAKE_SOURCE_DIR}/cuew/src/cuew.c"
)

# [Executable] Lucia
add_executable ( unit_test
    ${TEST_SOURCE}
)

# Increase warning level.
if (MSVC)
    target_compile_options(unit_test PRIVATE /W4)
else()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATU ${CMAKE_CXX_COMPILER_ID})
        target_compile_options(unit_test PRIVATE -fno-operator-names)
    endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(unit_test PRIVATE -Weverything -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation)
    endif ()
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} EasyCL clew cuew)

target_link_libraries(unit_test ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
add_sanitizers(unit_test)
