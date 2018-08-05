cmake_minimum_required(VERSION 3.10
    FATAL_ERROR)
project(ToribashPaper)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/paper.pdf
    COMMAND pdflatex ${CMAKE_CURRENT_SOURCE_DIR}/paper.ltx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/paper.ltx
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(toribash_paper ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/paper.pdf
)
