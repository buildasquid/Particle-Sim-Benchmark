find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} /usr/include/eigen3)

add_executable(cpu_sph src/cpu_sph.cpp)

target_link_libraries(cpu_sph ${OPENGL_gl_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_glu_LIBRARY})
