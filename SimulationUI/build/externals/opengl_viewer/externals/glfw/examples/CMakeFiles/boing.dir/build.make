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

# Include any dependencies generated for this target.
include externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/compiler_depend.make

# Include the progress variables for this target.
include externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/progress.make

# Include the compile flags for this target's objects.
include externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/flags.make

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/flags.make
externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o: /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/examples/boing.c
externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o -MF CMakeFiles/boing.dir/boing.c.o.d -o CMakeFiles/boing.dir/boing.c.o -c /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/examples/boing.c

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/boing.dir/boing.c.i"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/examples/boing.c > CMakeFiles/boing.dir/boing.c.i

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/boing.dir/boing.c.s"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/examples/boing.c -o CMakeFiles/boing.dir/boing.c.s

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/flags.make
externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o: /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/deps/glad.c
externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o -MF CMakeFiles/boing.dir/__/deps/glad.c.o.d -o CMakeFiles/boing.dir/__/deps/glad.c.o -c /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/deps/glad.c

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/boing.dir/__/deps/glad.c.i"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/deps/glad.c > CMakeFiles/boing.dir/__/deps/glad.c.i

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/boing.dir/__/deps/glad.c.s"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/deps/glad.c -o CMakeFiles/boing.dir/__/deps/glad.c.s

# Object files for target boing
boing_OBJECTS = \
"CMakeFiles/boing.dir/boing.c.o" \
"CMakeFiles/boing.dir/__/deps/glad.c.o"

# External object files for target boing
boing_EXTERNAL_OBJECTS =

externals/opengl_viewer/externals/glfw/examples/boing: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/boing.c.o
externals/opengl_viewer/externals/glfw/examples/boing: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/__/deps/glad.c.o
externals/opengl_viewer/externals/glfw/examples/boing: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/build.make
externals/opengl_viewer/externals/glfw/examples/boing: externals/opengl_viewer/externals/glfw/src/libglfw3.a
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/librt.a
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libm.so
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libX11.so
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libXrandr.so
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libXinerama.so
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libXxf86vm.so
externals/opengl_viewer/externals/glfw/examples/boing: /usr/lib/x86_64-linux-gnu/libXcursor.so
externals/opengl_viewer/externals/glfw/examples/boing: externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable boing"
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/boing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/build: externals/opengl_viewer/externals/glfw/examples/boing
.PHONY : externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/build

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/clean:
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples && $(CMAKE_COMMAND) -P CMakeFiles/boing.dir/cmake_clean.cmake
.PHONY : externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/clean

externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/depend:
	cd /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/externals/opengl_viewer/externals/glfw/examples /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples /home/lodeki/Documents/side_ups/LearningToFly/SimulationUI/build/externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : externals/opengl_viewer/externals/glfw/examples/CMakeFiles/boing.dir/depend

