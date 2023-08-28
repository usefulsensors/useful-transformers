find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(SOURCE_FILES
  attention.h
  bias.h
  matmul.h
  misc.h
  rknn_api.h
  rknn_matmul_api.h
  residual_attention_block.h
  softmax.h
  profiler.h
  conv.h
  layernorm.h
  attention.cc
  bias.cc
  matmul.cc
  residual_attention_block.cc
  softmax.cc
  profiler.cc
  conv.cc
  layernorm.cc
)

add_library(ctransformer SHARED STATIC ${SOURCE_FILES})
target_link_libraries(ctransformer
  PRIVATE
  "${PROJECT_SOURCE_DIR}/lib/librknnrt.so"
)

pybind11_add_module(pybind_residual_attention_block
  MODULE pybind_residual_attention_block.cc
)
pybind11_extension(pybind_residual_attention_block)
target_link_libraries(pybind_residual_attention_block
  PRIVATE ctransformer
)

install(FILES "${PROJECT_SOURCE_DIR}/lib/librknnrt.so" DESTINATION .)
