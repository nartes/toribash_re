cmake_minimum_required(VERSION 3.10
    FATAL_ERROR)
project(ToribashPaper)

function(add_ltx target_name)
  add_custom_command(
      OUTPUT
          ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.pdf
      COMMAND pdflatex ${CMAKE_CURRENT_SOURCE_DIR}/${target_name}.ltx
      DEPENDS
          ${CMAKE_CURRENT_SOURCE_DIR}/${target_name}.ltx
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${target_name} ALL
      DEPENDS
          ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.pdf
  )
endfunction()

add_ltx(ml-paper)
add_ltx(r2con-2018-slides)
