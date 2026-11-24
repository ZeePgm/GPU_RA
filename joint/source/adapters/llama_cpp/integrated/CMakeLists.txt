cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/gpu_configuration.cmake")
project(realgpu_llama LANGUAGES C CXX CUDA)
# Localized MSVC dependency output must match Ninja's UTF-8 prefix.
if(MSVC AND CMAKE_CXX_CL_SHOWINCLUDES_PREFIX MATCHES "娉ㄦ剰")
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
get_filename_component(LAB_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(LLAMA_SOURCE "${LAB_ROOT}/third_party/llama.cpp-integrated")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(HBFLAB_EVENTLAB_SOURCE "${LAB_ROOT}/third_party/HBF-EventLab" CACHE PATH "EventLab source checkout or verified archive")
if(NOT EXISTS "${HBFLAB_EVENTLAB_SOURCE}/include/hbf_eventlab/runtime/device_driver.hpp")
  message(FATAL_ERROR "Set HBFLAB_EVENTLAB_SOURCE to the HBF-EventLab source directory")
endif()
add_subdirectory("${HBFLAB_EVENTLAB_SOURCE}" eventlab)
if(MSVC)
  set_property(TARGET hbf_eventlab PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/EHsc>")
endif()
find_package(CUDAToolkit REQUIRED)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper.ptx"
 COMMAND "${CMAKE_CUDA_COMPILER}" --ptx --relocatable-device-code=true -std=c++17 -arch=${HBFLAB_PTX_ARCHITECTURE} "-I${LAB_ROOT}/include" "${LAB_ROOT}/src/async_ptx_helper.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper.ptx"
 DEPENDS "${LAB_ROOT}/src/async_ptx_helper.cu" "${LAB_ROOT}/include/hbflab/async_device.cuh" "${LAB_ROOT}/include/hbflab/async_protocol.hpp" "${LAB_ROOT}/include/hbflab/device.cuh" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper_embedded.hpp"
 COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper.ptx" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper_embedded.hpp" -DSYMBOL=embedded_async_ptx_helper -P "${LAB_ROOT}/cmake/embed_ptx.cmake"
 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper.ptx" "${LAB_ROOT}/cmake/embed_ptx.cmake" VERBATIM)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel.ptx"
 COMMAND "${CMAKE_CUDA_COMPILER}" --ptx  -std=c++17 -arch=${HBFLAB_PTX_ARCHITECTURE}  "${LAB_ROOT}/adapters/llama_cpp/integrated_weight_mul.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel.ptx"
 DEPENDS "${LAB_ROOT}/adapters/llama_cpp/integrated_weight_mul.cu" "${LAB_ROOT}/include/hbflab/async_device.cuh" "${LAB_ROOT}/include/hbflab/async_protocol.hpp" "${LAB_ROOT}/include/hbflab/device.cuh" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp"
 COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/weight_kernel.ptx" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp" -DSYMBOL=embedded_weight_kernel -P "${LAB_ROOT}/cmake/embed_ptx.cmake"
 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel.ptx" "${LAB_ROOT}/cmake/embed_ptx.cmake" VERBATIM)

add_library(realgpu_runtime STATIC "${LAB_ROOT}/src/backend.cpp" "${LAB_ROOT}/src/async_runtime.cu" "${LAB_ROOT}/src/live_runtime.cpp" "${LAB_ROOT}/src/capacity.cpp" "${LAB_ROOT}/src/page_file.cpp" "${LAB_ROOT}/src/environment.cpp" "${LAB_ROOT}/src/kv.cpp" "${LAB_ROOT}/src/capacity_gpu.cpp" "${LAB_ROOT}/src/async_module.cpp" "${LAB_ROOT}/src/ptx.cpp" "${LAB_ROOT}/src/sha256.cpp" "${CMAKE_CURRENT_BINARY_DIR}/async_ptx_helper_embedded.hpp")
target_include_directories(realgpu_runtime PRIVATE "${LLAMA_SOURCE}/vendor")
target_include_directories(realgpu_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" PUBLIC "${LAB_ROOT}/include")
target_link_libraries(realgpu_runtime PUBLIC HBFEventLab::core CUDA::cudart CUDA::cuda_driver)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)
foreach(flag COMMON TESTS TOOLS EXAMPLES SERVER APP)
  set(LLAMA_BUILD_${flag} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMIT "b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9" CACHE STRING "" FORCE)
add_subdirectory("${LLAMA_SOURCE}" llama)
add_custom_target(realgpu_kernel DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp")
add_dependencies(ggml-cuda realgpu_kernel)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp" TARGET_DIRECTORY ggml-cuda PROPERTIES GENERATED TRUE)
target_sources(ggml-cuda PRIVATE "${LAB_ROOT}/adapters/llama_cpp/integrated_norm_explicit.cu" "${LAB_ROOT}/adapters/llama_cpp/joint_weights.cu" "${LAB_ROOT}/weights/windowed_read/window.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_ptx.cpp" "${LAB_ROOT}/adapters/llama_cpp/integrated_profile.cpp" "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp")
set_source_files_properties("${LAB_ROOT}/adapters/llama_cpp/integrated_weights.cu" TARGET_DIRECTORY ggml-cuda PROPERTIES INCLUDE_DIRECTORIES "${LLAMA_SOURCE}/ggml/src/ggml-cuda")
target_include_directories(ggml-cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor")
target_compile_definitions(ggml-cuda PRIVATE HBFLAB_M2_BUILD)
target_sources(ggml-cuda PRIVATE "${LAB_ROOT}/adapters/llama_cpp/restored_routes.cpp")
target_link_libraries(ggml-cuda PRIVATE realgpu_runtime HBFEventLab::tensor_bridge)
find_path(M2_CUPTI_INCLUDE cupti.h HINTS "${CUDAToolkit_ROOT}/include" "${CUDAToolkit_ROOT}/extras/CUPTI/include" REQUIRED)
find_library(M2_CUPTI_LIBRARY NAMES cupti HINTS "${CUDAToolkit_ROOT}/lib" "${CUDAToolkit_ROOT}/lib/x64" "${CUDAToolkit_ROOT}/lib64" "${CUDAToolkit_ROOT}/extras/CUPTI/lib64" "${CUDAToolkit_ROOT}/extras/CUPTI/lib/x64" REQUIRED)
add_executable(realgpu_llama "${LAB_ROOT}/adapters/llama_cpp/integrated_runner.cpp")
target_include_directories(realgpu_llama PRIVATE "${HBFLAB_EVENTLAB_SOURCE}/include" "${LAB_ROOT}/include" "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor" "${M2_CUPTI_INCLUDE}")
target_link_libraries(realgpu_llama PRIVATE llama ggml-cuda CUDA::cudart ${M2_CUPTI_LIBRARY})
target_compile_definitions(realgpu_llama PRIVATE M0_LLAMA_COMMIT="${LLAMA_BUILD_COMMIT}")
if(MSVC)
  target_compile_options(realgpu_llama PRIVATE /utf-8 /W4)
endif()

add_executable(realgpu_attention_test "${LAB_ROOT}/adapters/llama_cpp/integrated_attention_test.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_profile.cpp")
target_include_directories(realgpu_attention_test PRIVATE "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor")
target_compile_definitions(realgpu_attention_test PRIVATE HBFLAB_M2_BUILD)
target_link_libraries(realgpu_attention_test PRIVATE realgpu_runtime ggml-base CUDA::cudart)

add_executable(realgpu_ptx_test "${LAB_ROOT}/adapters/llama_cpp/integrated_ptx_test.cpp" "${CMAKE_CURRENT_BINARY_DIR}/weight_kernel_embedded.hpp")
add_executable(realgpu_readonly_test "${LAB_ROOT}/tests/readonly_access.cpp")
target_link_libraries(realgpu_readonly_test PRIVATE realgpu_runtime)
add_dependencies(realgpu_ptx_test realgpu_kernel)
target_include_directories(realgpu_ptx_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${LLAMA_SOURCE}/vendor")
target_link_libraries(realgpu_ptx_test PRIVATE realgpu_runtime CUDA::cudart)

option(HBFLAB_K1_TESTS "Build K1 GPU transaction validation" OFF)
if(HBFLAB_K1_TESTS)
add_executable(hbflab_k1_test "${LAB_ROOT}/tests/kv_gpu_write.cu")
target_link_libraries(hbflab_k1_test PRIVATE realgpu_runtime CUDA::cudart)
add_executable(hbflab_k1_transactions "${LAB_ROOT}/tests/kv_transaction.cpp")
target_link_libraries(hbflab_k1_transactions PRIVATE realgpu_runtime CUDA::cudart)
add_executable(hbflab_k1_capacity "${LAB_ROOT}/tests/capacity_gpu.cu")
target_link_libraries(hbflab_k1_capacity PRIVATE realgpu_runtime CUDA::cudart)
endif()

add_executable(hbflab_k3_attention "${LAB_ROOT}/adapters/llama_cpp/k3_attention_test.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_profile.cpp")
target_include_directories(hbflab_k3_attention PRIVATE "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor")
target_compile_definitions(hbflab_k3_attention PRIVATE HBFLAB_M2_BUILD)
target_link_libraries(hbflab_k3_attention PRIVATE realgpu_runtime ggml-base CUDA::cudart)

add_executable(hbflab_k3_mixed "${LAB_ROOT}/adapters/llama_cpp/k3_mixed_test.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_profile.cpp")
target_include_directories(hbflab_k3_mixed PRIVATE "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor")
target_compile_definitions(hbflab_k3_mixed PRIVATE HBFLAB_M2_BUILD)
target_link_libraries(hbflab_k3_mixed PRIVATE realgpu_runtime ggml-base CUDA::cudart)

add_executable(hbflab_k3_capacity "${LAB_ROOT}/tests/k3.cpp")
target_link_libraries(hbflab_k3_capacity PRIVATE realgpu_runtime)

# Historical runners build the original target and invoke its filename directly.
function(add_legacy_executable legacy current)
  set(output_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  if(CMAKE_CONFIGURATION_TYPES)
    string(APPEND output_dir "/$<CONFIG>")
  endif()
  set(legacy_file "${output_dir}/${legacy}${CMAKE_EXECUTABLE_SUFFIX}")
  add_custom_command(OUTPUT "${legacy_file}"
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:${current}>" "${legacy_file}"
    DEPENDS ${current}
    VERBATIM)
  add_custom_target(${legacy} ALL DEPENDS "${legacy_file}")
endfunction()

add_executable(weight_read_window_check "${LAB_ROOT}/weights/windowed_read/fixture.cpp" "${LAB_ROOT}/weights/windowed_read/window.cu")
target_include_directories(weight_read_window_check PRIVATE "${LLAMA_SOURCE}/vendor")
target_link_libraries(weight_read_window_check PRIVATE realgpu_runtime CUDA::cudart)
add_legacy_executable(joint_b2_fixture weight_read_window_check)
add_executable(joint_media_gpu "${LAB_ROOT}/tests/joint_media_gpu.cpp")
target_link_libraries(joint_media_gpu PRIVATE realgpu_runtime CUDA::cudart)

add_executable(weight_access_model "${LAB_ROOT}/adapters/llama_cpp/weight_access_model.cpp")
target_include_directories(weight_access_model PRIVATE "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor" "${M2_CUPTI_INCLUDE}" "${LAB_ROOT}/include")
target_link_libraries(weight_access_model PRIVATE llama ggml-cuda CUDA::cudart ${M2_CUPTI_LIBRARY})
add_legacy_executable(five_model weight_access_model)
set(HBFLAB_Q_AUTO_LAUNCHER "" CACHE FILEPATH "Experimental compiler launcher for Q access tracing")
set(HBFLAB_FIVE_MEDIA_LAUNCHER "" CACHE FILEPATH "Experimental compiler launcher for five-weight live timing")
set(HBFLAB_ALL_MEDIA_LAUNCHER "" CACHE FILEPATH "Experimental compiler launcher for all-weight live timing")
if(HBFLAB_ALL_MEDIA_LAUNCHER)
  if(HBFLAB_Q_AUTO_LAUNCHER OR HBFLAB_FIVE_MEDIA_LAUNCHER)
    message(FATAL_ERROR "Choose a single experimental PTX launcher")
  endif()
  set(HBFLAB_FIVE_MEDIA_LAUNCHER "${HBFLAB_ALL_MEDIA_LAUNCHER}")
  target_compile_definitions(ggml-cuda PRIVATE HBFLAB_ALL_MEDIA_BUILD=1)
  target_compile_definitions(weight_access_model PRIVATE HBFLAB_ALL_MEDIA_BUILD=1)
  target_link_libraries(weight_access_model PRIVATE realgpu_runtime)
endif()
if(HBFLAB_Q_AUTO_LAUNCHER AND HBFLAB_FIVE_MEDIA_LAUNCHER)
  message(FATAL_ERROR "Choose a single experimental PTX launcher")
endif()
if(HBFLAB_Q_AUTO_LAUNCHER)
  target_sources(ggml-cuda PRIVATE "${LAB_ROOT}/adapters/llama_cpp/q_auto_device.cu")
  set_property(TARGET ggml-cuda PROPERTY CUDA_COMPILER_LAUNCHER
    "${Python3_EXECUTABLE};${HBFLAB_Q_AUTO_LAUNCHER}")
  get_filename_component(q_auto_dir "${HBFLAB_Q_AUTO_LAUNCHER}" DIRECTORY)
  foreach(unit convert mmvf)
    set_property(SOURCE "${LLAMA_SOURCE}/ggml/src/ggml-cuda/${unit}.cu"
      TARGET_DIRECTORY ggml-cuda APPEND PROPERTY OBJECT_DEPENDS
      "${HBFLAB_Q_AUTO_LAUNCHER}" "${q_auto_dir}/rewrite_ptx.py")
  endforeach()
  target_compile_definitions(weight_access_model PRIVATE HBFLAB_Q_AUTO_TRACE=1)
endif()
if(HBFLAB_FIVE_MEDIA_LAUNCHER)
  target_sources(ggml-cuda PRIVATE "${LAB_ROOT}/adapters/llama_cpp/weight_page_service.cu")
  set_property(TARGET ggml-cuda PROPERTY CUDA_COMPILER_LAUNCHER
    "${Python3_EXECUTABLE};${HBFLAB_FIVE_MEDIA_LAUNCHER}")
  get_filename_component(five_media_dir "${HBFLAB_FIVE_MEDIA_LAUNCHER}" DIRECTORY)
  foreach(unit convert mmvf binbcast getrows)
    set_property(SOURCE "${LLAMA_SOURCE}/ggml/src/ggml-cuda/${unit}.cu"
      TARGET_DIRECTORY ggml-cuda APPEND PROPERTY OBJECT_DEPENDS
      "${HBFLAB_FIVE_MEDIA_LAUNCHER}" "${five_media_dir}/rewrite_ptx.py"
      "${five_media_dir}/../v3-q-auto-001/compile_ptx.py"
      "${five_media_dir}/../v3-q-auto-001/rewrite_ptx.py")
  endforeach()
  if(HBFLAB_ALL_MEDIA_LAUNCHER)
    foreach(unit convert mmvf binbcast getrows)
      set_property(SOURCE "${LLAMA_SOURCE}/ggml/src/ggml-cuda/${unit}.cu"
        TARGET_DIRECTORY ggml-cuda APPEND PROPERTY OBJECT_DEPENDS
        "${five_media_dir}/../v3-five-timed-001/compile_five.py"
        "${five_media_dir}/../v3-five-timed-001/rewrite_ptx.py")
    endforeach()
  endif()
  target_compile_definitions(ggml-cuda PRIVATE HBFLAB_FIVE_MEDIA_BUILD=1)
  target_compile_definitions(weight_access_model PRIVATE HBFLAB_FIVE_MEDIA_BUILD=1)
endif()
add_executable(five_weight "${LAB_ROOT}/adapters/llama_cpp/five_weight.cpp")
target_include_directories(five_weight PRIVATE "${LAB_ROOT}/adapters/llama_cpp" "${LLAMA_SOURCE}/vendor")
target_link_libraries(five_weight PRIVATE llama ggml-cuda realgpu_runtime CUDA::cudart)

set(R2_DIR "${LAB_ROOT}/experiments/readonly_r2")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/down.ptx"
 COMMAND "${CMAKE_CUDA_COMPILER}" --ptx -std=c++17 -arch=${HBFLAB_PTX_ARCHITECTURE}
 "-I${LAB_ROOT}/include" "${R2_DIR}/down_probe.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/down.ptx"
 DEPENDS "${R2_DIR}/down_probe.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_down_scalar.cu" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/r2_half_to_float.ptx"
 COMMAND "${CMAKE_CUDA_COMPILER}" --ptx -std=c++17 -arch=${HBFLAB_PTX_ARCHITECTURE}
 "${LAB_ROOT}/adapters/llama_cpp/r2_half_to_float.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/r2_half_to_float.ptx"
 DEPENDS "${LAB_ROOT}/adapters/llama_cpp/r2_half_to_float.cu" VERBATIM)
foreach(kernel down r2_half_to_float)
 add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.hpp"
  COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/${kernel}.ptx"
  "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.hpp" "-DSYMBOL=embedded_${kernel}"
  -P "${LAB_ROOT}/cmake/embed_ptx.cmake"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.ptx" "${LAB_ROOT}/cmake/embed_ptx.cmake" VERBATIM)
endforeach()
add_custom_target(r2_consumers DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/down_embedded.hpp"
 "${CMAKE_CURRENT_BINARY_DIR}/r2_half_to_float_embedded.hpp")
add_dependencies(ggml-cuda r2_consumers)
target_compile_definitions(ggml-cuda PRIVATE HBFLAB_RESTORED_R2)
add_executable(r2_down "${R2_DIR}/down_runner.cpp" "${R2_DIR}/down_explicit.cu"
 "${R2_DIR}/down_timed.cu" "${CMAKE_CURRENT_BINARY_DIR}/down.ptx")
target_include_directories(r2_down PRIVATE "${LLAMA_SOURCE}/vendor")
target_link_libraries(r2_down PRIVATE realgpu_runtime)

add_executable(symbolic_f "${LAB_ROOT}/experiments/gpu_s/native_s8.cpp")
target_compile_definitions(symbolic_f PRIVATE GPU_S_JOINT GPU_S_TRANSFER_TRACE GPU_S_SEGMENTS GPU_FFN_EXPERIMENT)
target_include_directories(symbolic_f PRIVATE "${LLAMA_SOURCE}/vendor" "${LAB_ROOT}/include" "${M2_CUPTI_INCLUDE}")
target_link_libraries(symbolic_f PRIVATE llama ggml HBFEventLab::tensor_bridge CUDA::cudart ${M2_CUPTI_LIBRARY})
