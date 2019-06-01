cmake_minimum_required( VERSION 3.10 )

# project name and version
project( isai-kohris )

# setting default output directories for built targets
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )

#compiler settings
add_compile_options( -std=c++17 )
#add_compile_options( -stdlib=libc++ )
add_compile_options( -pthread )
add_compile_options( -Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual -Wno-noexcept-type -Wpedantic )
add_compile_options( -Weverything -Wno-float-equal -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-documentation -Wno-reserved-id-macro -Wno-switch-enum -Wno-weak-vtables -Wno-missing-prototypes -Wno-padded -Wno-missing-noreturn -Wno-exit-time-destructors -Wno-documentation-unknown-command -Wno-unused-template -Wno-undef -Wno-format-nonliteral -Wno-global-constructors )
add_compile_options( -g )
add_definitions( -DDEBUG )
add_definitions( -D_DEBUG )
add_compile_options( -O2 )

# exporting of llvm compiler_commands.json enabled
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# project executable target
add_executable( kohris
    src/prng.h
    src/prng.cpp
    src/dataset.h
    src/dataset.cpp
    src/kohnet.h
    src/kohnet.cpp
    src/test.cpp )

target_include_directories( kohris
  PRIVATE
    src )
