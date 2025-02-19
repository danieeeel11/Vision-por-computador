cmake_minimum_required(VERSION 2.8)

project(Parcial2)

# Incluir la biblioteca pthread
find_package(Threads REQUIRED)

include_directories("/usr/local/include")
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries( ${PROJECT_NAME}  ${OpenCV_LIBS} Threads::Threads )
