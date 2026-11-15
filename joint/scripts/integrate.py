"""Apply the reviewed joint adapter changes only to the independent copy."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'source'
def replace(name,old,new):
    p=ROOT/name;s=p.read_text(encoding='utf-8');assert old in s,(name,old[:80]);p.write_text(s.replace(old,new),encoding='utf-8')
def main():
    replace('include/hbflab/backend.hpp','    std::uint64_t reserve(std::uint64_t page_aligned_bytes);','    std::uint64_t reserve(std::uint64_t page_aligned_bytes);\n    void reserve_gap(std::uint64_t bytes);')
    replace('src/backend.cpp','  std::vector<std::array<std::uint64_t, 64>> contents;','  std::vector<std::array<std::uint64_t, 64>> contents;\n  std::vector<bool> initialized;')
    replace('src/backend.cpp','if (at < allocated)','if (at < allocated && initialized.at(at/4096))')
    replace('src/backend.cpp','u.logical>=allocated || block_base','u.logical>=allocated || !initialized.at(u.logical/4096) || block_base')
    replace('src/backend.cpp','        return false;\n    }\n    return bytes <= max_read_bytes;', '        return false;\n      for(auto p=span.logical/4096;p<=(span.logical+span.bytes-1)/4096;++p)\n        if(!initialized.at(p))return false;\n    }\n    return bytes <= max_read_bytes;')
    replace('src/backend.cpp','      s.contents.push_back({});','      s.contents.push_back({});\n      s.initialized.push_back(true);')
    replace('src/backend.cpp','std::uint64_t TimingBackend::reserve(std::uint64_t bytes) {', '''void TimingBackend::reserve_gap(std::uint64_t bytes) {
  auto& s=*impl_;
  require(!s.streaming && !s.poisoned && !s.stats.input_closed,"gap reservation unavailable");
  const auto group=std::uint64_t(s.options.geometry.block_group_dlus())*4096;
  if(!bytes || bytes%group || s.allocated%group || bytes>s.options.geometry.channel_bytes()-s.allocated)
    throw std::invalid_argument("gap must preserve whole block groups and fit capacity");
  s.contents.resize((s.allocated+bytes)/4096);
  s.initialized.resize((s.allocated+bytes)/4096,false);
  s.allocated+=bytes;
}
std::uint64_t TimingBackend::reserve(std::uint64_t bytes) {''')
    replace('src/backend.cpp','logical % 4096 || logical >= s.allocated)', 'logical % 4096 || logical >= s.allocated || !s.initialized.at(logical/4096))')
    replace('include/hbflab/async_runtime.hpp','    BackendOptions backend{};','    BackendOptions backend{};\n    std::vector<std::uint64_t> readonly_bases;\n    std::uint64_t readonly_layout_bytes=0;')
    replace('src/async_runtime.cu','    std::uint64_t wall_epoch=0,dispatch_serial=0;','    std::uint64_t wall_epoch=0,dispatch_serial=0,registered_end=0;')
    replace('src/async_runtime.cu','    try{s.ranges.push_back({address,bytes,s.backend.reserve(padded)});}', '''    try{
        if(!s.options.readonly_bases.empty()) {
            if(s.ranges.size()>=s.options.readonly_bases.size())throw std::invalid_argument("missing readonly base");
            const auto base=s.options.readonly_bases.at(s.ranges.size());
            if(base<s.registered_end)throw std::invalid_argument("readonly logical overlap");
            if(base>s.registered_end)s.backend.reserve_gap(base-s.registered_end);
            s.registered_end=base;
        }
        const auto logical=s.backend.reserve(padded);
        s.ranges.push_back({address,bytes,logical});s.registered_end=logical+padded;
    }''')
    replace('src/async_runtime.cu','    const auto base=s.backend.reserve(bytes);s.media_ranges.push_back({base,bytes});return base;', '''    if(s.options.readonly_layout_bytes<s.registered_end && s.options.readonly_layout_bytes)
        throw std::invalid_argument("readonly layout too small");
    if(s.options.readonly_layout_bytes>s.registered_end) {
        s.backend.reserve_gap(s.options.readonly_layout_bytes-s.registered_end);
        s.registered_end=s.options.readonly_layout_bytes;
    }
    const auto base=s.backend.reserve(bytes);s.registered_end=base+bytes;s.media_ranges.push_back({base,bytes});return base;''')
    replace('src/async_runtime.cu','s.poisoned || s.ranges.empty() || shapes.empty()', 's.poisoned || (s.ranges.empty() && s.media_ranges.empty()) || shapes.empty()')
    replace('src/async_runtime.cu','        ck(cudaMalloc(reinterpret_cast<void**>(&s.device_ranges),s.ranges.size()*sizeof(Range)));\n        ck(cudaMemcpy(s.device_ranges,s.ranges.data(),s.ranges.size()*sizeof(Range),cudaMemcpyHostToDevice));', '''        if(!s.ranges.empty()) {
            ck(cudaMalloc(reinterpret_cast<void**>(&s.device_ranges),s.ranges.size()*sizeof(Range)));
            ck(cudaMemcpy(s.device_ranges,s.ranges.data(),s.ranges.size()*sizeof(Range),cudaMemcpyHostToDevice));
        }''')
    replace('adapters/llama_cpp/instrumented_bridge.hpp','HBFLAB_M2_API void hbflab_m2_bind(', 'HBFLAB_M2_API void hbflab_joint_bind(const ggml_tensor* const*,unsigned);\nHBFLAB_M2_API void hbflab_m2_bind(')
    replace('adapters/llama_cpp/integrated_runner.cpp','hbflab::platform::set_environment("HBFLAB_M4_PAGED",std::string(argv[6])=="native"?"0":"1");', '''const char* kv_setting=std::getenv("HBFLAB_JOINT_KV");
    if(!kv_setting || (std::string(kv_setting)!="native" && std::string(kv_setting)!="paged"))throw std::runtime_error("explicit joint KV setting required");
    hbflab::platform::set_environment("HBFLAB_M4_PAGED",std::string(kv_setting)=="paged"?"1":"0");''')
    replace('adapters/llama_cpp/integrated_runner.cpp','if (observe || std::string(argv[6])!="native") { cp.cb_eval = observe_node; cp.cb_eval_user_data = nullptr; }', 'cp.cb_eval = observe_node; cp.cb_eval_user_data = nullptr;')
    replace('adapters/llama_cpp/integrated_runner.cpp','    hbflab_m2_bind(hbflab_m2_find_tensor(model.get(),profile.norm_name.c_str()), hbflab_m2_find_tensor(model.get(),profile.down_name.c_str()));', '''    const char* scale_setting=std::getenv("HBFLAB_JOINT_SCALE");
    const std::string scale=scale_setting?scale_setting:"";
    if(scale!="F1" && scale!="F2")throw std::runtime_error("joint scale must be F1 or F2");
    std::vector<const ggml_tensor*> selected;
    const std::vector<int> selected_layers=scale=="F1"?std::vector<int>{10}:std::vector<int>{0,10,20,27};
    for(int layer:selected_layers)for(const char* role:{"gate","up","down"}){
        const auto name="blk."+std::to_string(layer)+".ffn_"+role+".weight";
        selected.push_back(hbflab_m2_find_tensor(model.get(),name.c_str()));
    }
    hbflab_joint_bind(selected.data(),static_cast<unsigned>(selected.size()));
    if(n!=44 || count!=8 || sequence_count!=1)throw std::runtime_error("joint scope requires 44 input and 8 output tokens, one sequence");''')
    replace('adapters/llama_cpp/integrated/CMakeLists.txt','"${LAB_ROOT}/adapters/llama_cpp/integrated_weights.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu"', '"${LAB_ROOT}/adapters/llama_cpp/joint_weights.cu" "${LAB_ROOT}/experiments/readonly_b2/window.cu" "${LAB_ROOT}/adapters/llama_cpp/integrated_kv.cu"')
    # Old PTX hooks are retained as built code but the joint adapter never calls them.
    print('Integrated source changes applied')
if __name__=='__main__': main()
