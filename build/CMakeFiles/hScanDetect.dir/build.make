# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhy/project/dg270Test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhy/project/dg270Test/build

# Include any dependencies generated for this target.
include CMakeFiles/hScanDetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hScanDetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hScanDetect.dir/flags.make

CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o: CMakeFiles/hScanDetect.dir/flags.make
CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o: hScanDetect_autogen/mocs_compilation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhy/project/dg270Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o -c /home/zhy/project/dg270Test/build/hScanDetect_autogen/mocs_compilation.cpp

CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhy/project/dg270Test/build/hScanDetect_autogen/mocs_compilation.cpp > CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.i

CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhy/project/dg270Test/build/hScanDetect_autogen/mocs_compilation.cpp -o CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.s

CMakeFiles/hScanDetect.dir/main.cpp.o: CMakeFiles/hScanDetect.dir/flags.make
CMakeFiles/hScanDetect.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhy/project/dg270Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/hScanDetect.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hScanDetect.dir/main.cpp.o -c /home/zhy/project/dg270Test/main.cpp

CMakeFiles/hScanDetect.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hScanDetect.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhy/project/dg270Test/main.cpp > CMakeFiles/hScanDetect.dir/main.cpp.i

CMakeFiles/hScanDetect.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hScanDetect.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhy/project/dg270Test/main.cpp -o CMakeFiles/hScanDetect.dir/main.cpp.s

# Object files for target hScanDetect
hScanDetect_OBJECTS = \
"CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o" \
"CMakeFiles/hScanDetect.dir/main.cpp.o"

# External object files for target hScanDetect
hScanDetect_EXTERNAL_OBJECTS =

hScanDetect: CMakeFiles/hScanDetect.dir/hScanDetect_autogen/mocs_compilation.cpp.o
hScanDetect: CMakeFiles/hScanDetect.dir/main.cpp.o
hScanDetect: CMakeFiles/hScanDetect.dir/build.make
hScanDetect: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.12.8
hScanDetect: /usr/lib/x86_64-linux-gnu/libpython3.8.so
hScanDetect: CMakeFiles/hScanDetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhy/project/dg270Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable hScanDetect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hScanDetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hScanDetect.dir/build: hScanDetect

.PHONY : CMakeFiles/hScanDetect.dir/build

CMakeFiles/hScanDetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hScanDetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hScanDetect.dir/clean

CMakeFiles/hScanDetect.dir/depend:
	cd /home/zhy/project/dg270Test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhy/project/dg270Test /home/zhy/project/dg270Test /home/zhy/project/dg270Test/build /home/zhy/project/dg270Test/build /home/zhy/project/dg270Test/build/CMakeFiles/hScanDetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hScanDetect.dir/depend
