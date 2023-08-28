find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(SOURCE_FILES
  whisper.h
  whisper.cc
)
  
add_library(whisper SHARED STATIC ${SOURCE_FILES})
target_include_directories(whisper PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(whisper
  PRIVATE ctransformer
)

pybind11_add_module(pybind_whisper MODULE pybind_whisper.cc)
pybind11_extension(pybind_whisper)
target_include_directories(pybind_whisper PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(pybind_whisper
  PRIVATE whisper
)

install(TARGETS pybind_whisper DESTINATION .)
