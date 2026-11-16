cmake_minimum_required(VERSION 3.16)

project(ImageDenoiser
    VERSION 1.0.0
    DESCRIPTION "Modular, extensible image denoising library and CLI"
    LANGUAGES CXX)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Convenience for local MSYS2/MinGW-w64 setups: point CMake at the OpenCV
# package installed there so plain `cmake` (and IDEs like CLion) work without
# extra command-line flags. Search paths passed explicitly by the user always
# take precedence. Remove this block if you obtain OpenCV from another source
# (vcpkg, system packages, ...).
#
# Note: appended to CMAKE_PREFIX_PATH rather than cached in OpenCV_DIR so this
# also recovers cleanly from a previous configure that left OpenCV_DIR-NOTFOUND
# behind in the CMake cache.
set(_denoiser_msys_prefix "")
# 1) The MSYS2 environment the configured compiler belongs to, whatever the
#    install root (C:/msys64, D:/msys64, C:/mingw64, ...). The OpenCV built for
#    that exact toolchain is the one that must be linked and bundled, so it
#    wins over any other candidate. A non-MinGW compiler (MSVC) resolves to a
#    directory without an OpenCV config here and simply doesn't match.
if(CMAKE_CXX_COMPILER)
    get_filename_component(_denoiser_cc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_denoiser_cc_env "${_denoiser_cc_bin}" DIRECTORY)
    if(EXISTS "${_denoiser_cc_env}/lib/cmake/opencv4/OpenCVConfig.cmake")
        set(_denoiser_msys_prefix "${_denoiser_cc_env}")
    endif()
endif()
# 2) Otherwise, the standard install root, first environment that ships
#    OpenCV (covers e.g. a CLion-bundled compiler with MSYS2 OpenCV).
if(NOT _denoiser_msys_prefix)
    foreach(_env IN ITEMS mingw64 ucrt64 clang64 clangarm64)
        if(EXISTS "C:/msys64/${_env}/lib/cmake/opencv4/OpenCVConfig.cmake")
            set(_denoiser_msys_prefix "C:/msys64/${_env}")
            break()
        endif()
    endforeach()
endif()
if(_denoiser_msys_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_denoiser_msys_prefix}")
    message(STATUS "MSYS2/MinGW OpenCV auto-detected: ${_denoiser_msys_prefix}")
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(denoiser STATIC
    src/ImageDenoiser.cpp
    src/AnisotropicDiffusionFilter.cpp
    src/BilateralFilter.cpp
    src/NonLocalMeansFilter.cpp
    src/DctFilter.cpp
    src/TotalVariationFilter.cpp
    src/DenoiserFactory.cpp)
add_library(ImageDenoiser::Denoiser ALIAS denoiser)

target_compile_features(denoiser PUBLIC cxx_std_17)

target_include_directories(denoiser
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# OpenCV is used for image representation and I/O only; consumers of the
# library headers need it too, hence PUBLIC.
target_link_libraries(denoiser
    PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs)

# Optional: multithread the per-pixel filtering loops with OpenMP. Internal
# implementation detail only, so PRIVATE on both the link and the define.
if(OpenMP_CXX_FOUND)
    target_link_libraries(denoiser PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(denoiser PRIVATE DENOISER_WITH_OPENMP)
    message(STATUS "OpenMP found: parallel filtering loops enabled")
endif()

# ---------------------------------------------------------------------------
# Command-line application
# ---------------------------------------------------------------------------
add_executable(denoiser_cli
    src/main.cpp
    src/InteractiveSession.cpp)
add_executable(ImageDenoiser::CLI ALIAS denoiser_cli)
set_target_properties(denoiser_cli PROPERTIES OUTPUT_NAME denoiser-cli)
target_link_libraries(denoiser_cli PRIVATE ImageDenoiser::Denoiser)

# ---------------------------------------------------------------------------
# Windows runtime bundling
# ---------------------------------------------------------------------------
# The MinGW/OpenCV DLLs live in the toolchain's bin directory, which is usually
# not on a user's PATH. After each build, resolve every transitive DLL
# dependency of the CLI and copy it next to the executable, so the binary runs
# standalone (double-click, CLion, plain terminal).
if(WIN32)
    option(DENOISER_BUNDLE_RUNTIME
        "Copy the CLI's runtime DLLs next to the executable after building" ON)

    if(DENOISER_BUNDLE_RUNTIME)
        # Directories that may contain the DLLs: the OpenCV bin directory (via
        # the imported targets' runtime locations) and the compiler's bin
        # directory (libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll).
        set(_denoiser_dll_dirs "")
        foreach(_target opencv_core opencv_imgproc opencv_imgcodecs)
            get_target_property(_runtime "${_target}" IMPORTED_LOCATION_RELEASE)
            if(NOT _runtime)
                get_target_property(_runtime "${_target}" IMPORTED_LOCATION)
            endif()
            if(_runtime)
                get_filename_component(_dir "${_runtime}" DIRECTORY)
                list(APPEND _denoiser_dll_dirs "${_dir}")
            endif()
        endforeach()
        get_filename_component(_compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        list(APPEND _denoiser_dll_dirs "${_compiler_bin}")
        list(REMOVE_DUPLICATES _denoiser_dll_dirs)

        # PE import-table scanning needs binutils' objdump; locate it up front
        # (IDE-spawned builds may have a minimal PATH) and hand it to the script.
        find_program(_denoiser_objdump objdump HINTS "${_compiler_bin}")
        mark_as_advanced(_denoiser_objdump)

        add_custom_command(TARGET denoiser_cli POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
                    "-Dexecutable=$<TARGET_FILE:denoiser_cli>"
                    "-Dobjdump=${_denoiser_objdump}"
                    "-Ddll_dirs=${_denoiser_dll_dirs}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixupRuntime.cmake"
            VERBATIM)
    endif()
endif()

# ---------------------------------------------------------------------------
# Strict warnings and optimization
# ---------------------------------------------------------------------------
function(denoiser_apply_strict_flags target)
    if(MSVC)
        # MSVC equivalents of -Wall -Wextra -Werror -O3
        target_compile_options(${target}
            PRIVATE /W4 /permissive- /WX $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(${target}
            PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)
    endif()
endfunction()

denoiser_apply_strict_flags(denoiser)
denoiser_apply_strict_flags(denoiser_cli)

# ---------------------------------------------------------------------------
# Installation (the internal src/detail/ header is deliberately not shipped)
# ---------------------------------------------------------------------------
install(TARGETS denoiser denoiser_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
