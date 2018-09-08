cmake_minimum_required(VERSION 3.10
    FATAL_ERROR)

add_subdirectory(src/patch_toribash patch_toribash)
add_subdirectory(src/ddpg ddpg)
add_subdirectory(doc/ doc)
