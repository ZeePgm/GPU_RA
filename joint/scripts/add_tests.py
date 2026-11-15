from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1];S=R/'source'
shutil.copyfile(R/'scripts/joint_backend_test.cpp',S/'tests/joint_backend_test.cpp')
shutil.copyfile(R/'scripts/joint_media_gpu.cpp',S/'tests/joint_media_gpu.cpp')
shutil.copyfile(R.parents[1]/'experiments/k5_r2/feedback.cpp',S/'tests/joint_k5_feedback.cpp')
p=S/'CMakeLists.txt'
p.write_text(p.read_text(encoding='utf-8')+'''\nif(BUILD_TESTING)
  add_executable(joint_backend_test tests/joint_backend_test.cpp)
  target_link_libraries(joint_backend_test PRIVATE hbflab_backend)
  add_test(NAME lab_joint_backend COMMAND joint_backend_test)
  add_executable(joint_k5_feedback tests/joint_k5_feedback.cpp)
  target_link_libraries(joint_k5_feedback PRIVATE hbflab_backend)
  add_test(NAME lab_joint_k5_feedback COMMAND joint_k5_feedback)
endif()
''',encoding='utf-8')
p=S/'adapters/llama_cpp/integrated/CMakeLists.txt'
p.write_text(p.read_text(encoding='utf-8')+'''\nadd_executable(joint_b2_fixture "${LAB_ROOT}/experiments/readonly_b2/fixture.cpp" "${LAB_ROOT}/experiments/readonly_b2/window.cu")
target_include_directories(joint_b2_fixture PRIVATE "${LLAMA_SOURCE}/vendor")
target_link_libraries(joint_b2_fixture PRIVATE realgpu_runtime CUDA::cudart)
add_executable(joint_media_gpu "${LAB_ROOT}/tests/joint_media_gpu.cpp")
target_link_libraries(joint_media_gpu PRIVATE realgpu_runtime CUDA::cudart)
''',encoding='utf-8')
p=S/'experiments/readonly_b2/fixture.cpp'
p.write_text(p.read_text(encoding='utf-8').replace('o.backend.geometry={1,1,2,16,4096,1};','o.backend.geometry={1,1,2,4,4096,1};o.backend.direct_device_transactions=true;o.backend.coalesce_capacity_feedback=true;'),encoding='utf-8')
p=S/'adapters/llama_cpp/integrated_kv.cu'
s=p.read_text(encoding='utf-8');old='s.kv->read(seq,layer,token,kind,bytes.data(),row);content.append(bytes.data(),row);'
assert old in s
p.write_text(s.replace(old,'s.kv->read(seq,layer,token,kind,bytes.data(),row);s.drain();content.append(bytes.data(),row);'),encoding='utf-8')
print('Added joint CPU test and B2 GPU qualification on common geometry')
