#pragma once

#include "ggml-backend.h"
#ifdef HBFLAB_Q_AUTO_TRACE
#include "q_auto_device_api.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// The graph observer verifies which consumer executed and captures its output.
// Device instrumentation is selected only by the single registered address range.
class QAutoModel {
    using Json = nlohmann::ordered_json;
    const ggml_tensor *weight_ = nullptr;
    const ggml_tensor *pending_ = nullptr;
    std::filesystem::path directory_;
    bool trace_ = false;
    bool device_registered_ = false;
    int step_ = -1;
    bool captured_ = false;
    Json report_;
    Json row_;

    static void require(bool ok, const char *why) {
        if (!ok) throw std::runtime_error(why);
    }
    static void check_cuda(int status, const char *action) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(static_cast<cudaError_t>(status)));
        }
    }
    template <typename T>
    void write_array(const std::string &file, const std::vector<T> &values) const {
        std::ofstream out(directory_ / file, std::ios::binary);
        out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(T)));
        out.close();
        require(bool(out), "cannot write Q audit array");
    }
    void collect_counts() {
        std::vector<std::uint32_t> counts(ggml_nbytes(weight_) / sizeof(std::uint16_t));
        std::array<std::uint64_t, q_auto::summary_words> summary{};
        check_cuda(hbflab_q_auto_device_step_end(counts.data(), counts.size(), summary.data(), summary.size()), "Q trace read");
        std::vector<std::uint64_t> sites(q_auto::site_count);
        check_cuda(hbflab_q_auto_device_sites(sites.data(), sites.size()), "Q instruction site read");
        std::uint64_t total = 0, untouched = 0, multiple = 0;
        for (auto value : counts) {
            total += value;
            untouched += value == 0;
            multiple += value > 1;
        }
        const auto suffix = std::to_string(step_);
        const auto counts_file = "counts-" + suffix + ".u32";
        const auto sites_file = "site-" + suffix + ".u64";
        write_array(counts_file, counts);
        write_array(sites_file, sites);
        row_["counts_file"] = counts_file;
        row_["sites_file"] = sites_file;
        row_["words"] = counts.size();
        row_["untouched_words"] = untouched;
        row_["multiply_read_words"] = multiple;
        row_["total_word_reads"] = total;
        row_["device_summary"] = {{"convert_loads", summary[q_auto::convert_loads]},
            {"convert_bytes", summary[q_auto::convert_bytes]}, {"decode_loads", summary[q_auto::decode_loads]},
            {"decode_bytes", summary[q_auto::decode_bytes]}, {"partial_or_misaligned", summary[q_auto::partial_or_misaligned]}};
    }
public:
    QAutoModel() = default;
    QAutoModel(const QAutoModel &) = delete;
    QAutoModel &operator=(const QAutoModel &) = delete;
    ~QAutoModel() { if (device_registered_) hbflab_q_auto_device_finish(); }

    void bind(const std::string &mode, const ggml_tensor *weight, std::uint64_t logical_base,
              const std::filesystem::path &directory, int prompt_tokens, int outputs) {
        require(mode == "native" || mode == "trace", "Q auto mode must be native or trace");
        require(!weight_ && weight && std::string(weight->name) == "blk.10.attn_q.weight" &&
                weight->type == GGML_TYPE_F16 && weight->ne[0] == 1024 && weight->ne[1] == 2048 &&
                weight->ne[2] == 1 && weight->ne[3] == 1 && ggml_is_contiguous(weight) &&
                weight->data && ggml_nbytes(weight) == 4194304, "Q auto registered tensor identity");
        require(prompt_tokens == 44 && outputs >= 2, "Q audit requires 44-token prefill and decode");
        cudaPointerAttributes attributes{};
        check_cuda(cudaPointerGetAttributes(&attributes, weight->data), "Q weight CUDA identity");
        require(attributes.type == cudaMemoryTypeDevice && attributes.device == 0, "Q weight must be on CUDA0");
        weight_ = weight;
        trace_ = mode == "trace";
        directory_ = directory;
        if (trace_) {
            check_cuda(hbflab_q_auto_device_begin(weight->data, ggml_nbytes(weight)), "Q device registration");
            device_registered_ = true;
        }
        report_ = {{"mode", mode}, {"tensor", weight->name}, {"bytes", ggml_nbytes(weight)},
            {"logical_base", logical_base}, {"device_address", reinterpret_cast<std::uintptr_t>(weight->data)},
            {"cuda_device", attributes.device}, {"registrations", 1}, {"device_registrations", int(device_registered_)},
            {"steps", Json::array()}};
    }
    void begin_step(int step) {
        if (!weight_) return;
        require(step == step_ + 1 && !pending_, "Q auto step order");
        step_ = step;
        captured_ = false;
        row_ = {{"step", step}, {"phase", step == 0 ? "prefill" : "decode"}};
        if (trace_) check_cuda(hbflab_q_auto_device_step_begin(), "Q trace reset");
    }
    bool observe_before(const ggml_tensor *node) {
        if (!weight_ || std::find(std::begin(node->src), std::end(node->src), weight_) == std::end(node->src)) return false;
        require(step_ >= 0 && !pending_ && !captured_, "Q graph consumer repeated");
        const int columns = step_ == 0 ? 44 : 1;
        require(node->op == GGML_OP_MUL_MAT && node->src[0] == weight_ && node->src[1] &&
                node->src[1]->type == GGML_TYPE_F32 && node->src[1]->ne[0] == weight_->ne[0] &&
                node->src[1]->ne[1] == columns && node->type == GGML_TYPE_F32 && node->ne[0] == weight_->ne[1] &&
                node->ne[1] == columns && node->ne[2] == 1 && node->ne[3] == 1 && ggml_is_contiguous(node),
                "Q graph consumer op, input or output shape mismatch");
        row_["consumer"] = node->name;
        row_["op"] = ggml_op_name(node->op);
        row_["weight_slot"] = 0;
        row_["activation_columns"] = columns;
        pending_ = node;
        return true;
    }
    void observe_after(const ggml_tensor *node) {
        if (!weight_ || node != pending_) return;
        std::vector<float> values(std::size_t(ggml_nelements(node)));
        ggml_backend_tensor_get(node, values.data(), 0, values.size() * sizeof(float));
        const auto file = "q-output-" + std::to_string(step_) + ".f32";
        write_array(file, values);
        row_["projection_file"] = file;
        row_["projection_values"] = values.size();
        captured_ = true;
        pending_ = nullptr;
    }
    void end_step() {
        if (!weight_) return;
        require(captured_ && !pending_, "Q graph output missing");
        if (trace_) collect_counts();
        report_["steps"].push_back(row_);
    }
    void finish() {
        if (!weight_) return;
        if (device_registered_) {
            check_cuda(hbflab_q_auto_device_finish(), "Q device release");
            device_registered_ = false;
        }
        require(report_["steps"].size() >= 2, "Q prefill/decode coverage missing");
        report_["finished"] = true;
        std::ofstream out(directory_ / "q-auto.json");
        out << report_.dump(2) << '\n';
        out.close();
        require(bool(out), "cannot write Q run summary");
    }
};
#else
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
class QAutoModel {
public:
    void bind(const std::string &, const ggml_tensor *, std::uint64_t, const std::filesystem::path &, int, int) {
        throw std::runtime_error("the model binary was built without Q trace support");
    }
    void begin_step(int) {}
    bool observe_before(const ggml_tensor *) { return false; }
    void observe_after(const ggml_tensor *) {}
    void end_step() {}
    void finish() {}
};
#endif
