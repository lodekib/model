# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1381/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1381/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build

# Utility rule file for uninstall.

# Include any custom commands dependencies for this target.
include externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/compiler_depend.make

# Include the progress variables for this target.
include externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/progress.make

externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall:
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw && /snap/cmake/1381/bin/cmake -P /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/cmake_uninstall.cmake

uninstall: externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall
uninstall: externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/build.make
.PHONY : uninstall

# Rule to build all files generated by this target.
externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/build: uninstall
.PHONY : externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/build

externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/clean:
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw && $(CMAKE_COMMAND) -P CMakeFiles/uninstall.dir/cmake_clean.cmake
.PHONY : externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/clean

externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/depend:
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : externals/opengl_viewer/externals/glfw/CMakeFiles/uninstall.dir/depend
