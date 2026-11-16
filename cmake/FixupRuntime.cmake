# Script-mode helper (invoked by the CLI's POST_BUILD custom command).
#
# Resolves every transitive DLL dependency of the freshly linked CLI and copies
# it next to the executable, so the binary runs standalone on Windows without
# the toolchain's bin directory on PATH.
#
# Variables expected from the calling scope (-D ... -P FixupRuntime.cmake):
#   executable - absolute path to the CLI executable
#   objdump    - optional absolute path to binutils' objdump (dependency scanner)
#   dll_dirs   - semicolon-separated list of directories to search for DLLs

if(NOT executable)
    message(FATAL_ERROR "FixupRuntime.cmake: the 'executable' variable is required")
endif()

get_filename_component(executable_dir "${executable}" DIRECTORY)

# BundleUtilities' dependency scanner (GetPrerequisites) picks 'dumpbin' on
# Windows first and falls back to objdump. Force our explicitly located objdump
# so the scan works regardless of the invoking environment's PATH.
if(objdump)
    set(gp_tool "${objdump}")
else()
    # GetPrerequisites only warns and scans nothing when the scanner is
    # missing, silently producing an executable that cannot run. Degrade to a
    # bulk copy of everything in dll_dirs instead: imprecise, but the CLI
    # still runs standalone.
    foreach(_dir IN LISTS dll_dirs)
        file(GLOB _fixup_all_dlls "${_dir}/*.dll")
        if(_fixup_all_dlls)
            file(COPY ${_fixup_all_dlls} DESTINATION "${executable_dir}")
        endif()
    endforeach()
    return()
endif()

include(BundleUtilities)

fixup_bundle("${executable}" "" "${dll_dirs}")
