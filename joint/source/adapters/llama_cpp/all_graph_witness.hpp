#pragma once

#include "ggml-backend.h"
#include "hbflab/ptx.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Graph evidence is collected independently of the device range matcher.
// Each hash represents the complete logical F32 output in ggml element order.
class AllGraphWitness {
    using Json = nlohmann::ordered_json;
    std::map<const ggml_tensor *, std::size_t> ids_;
    std::set<const ggml_tensor *> consumed_;
    Json phases_ = Json::array();
    Json pending_ = Json::array();
    const ggml_tensor *pending_node_ = nullptr;
    std::filesystem::path directory_;

    static void check(bool value, const char *why) {
        if (!value) throw std::runtime_error(why);
    }
    static Json shape(const ggml_tensor *t) {
        return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    }
    static std::string output_hash(const ggml_tensor *node) {
        check(node->type == GGML_TYPE_F32, "expected F32 original graph output");
        std::string raw(ggml_nbytes(node), '\0');
        ggml_backend_tensor_get(node, raw.data(), 0, raw.size());
        if (ggml_is_contiguous(node)) return hbflab::sha256(raw);
        std::string packed(std::size_t(ggml_nelements(node)) * sizeof(float), '\0');
        std::size_t dest = 0;
        for (int64_t i3 = 0; i3 < node->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < node->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < node->ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                        const auto offset = i0 * node->nb[0] + i1 * node->nb[1] + i2 * node->nb[2] + i3 * node->nb[3];
                        check(offset <= raw.size() - sizeof(float), "invalid strided output bounds");
                        std::memcpy(packed.data() + dest, raw.data() + offset, sizeof(float));
                        dest += sizeof(float);
                    }
        return hbflab::sha256(packed);
    }
public:
    void start(const std::map<const ggml_tensor *, std::size_t> &ids, const std::filesystem::path &directory) {
        ids_ = ids;
        directory_ = directory;
    }
    void step_begin(int step) {
        if (ids_.empty()) return;
        check(std::size_t(step) == phases_.size() && !pending_node_, "all-weight graph phase order");
        consumed_.clear();
        phases_.push_back({{"step", step}, {"entries", Json::array()}});
    }
    void before(const ggml_tensor *node) {
        if (ids_.empty()) return;
        check(!pending_node_, "missing graph consumer completion");
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_PERMUTE) return;
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            const auto *source = node->src[slot];
            const auto *weight = source;
            while (weight && !ids_.count(weight)) weight = weight->view_src;
            if (!weight) continue;
            check(node->op == GGML_OP_MUL || node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_GET_ROWS,
                  "unexpected consuming operator for an original model weight");
            const auto *other = node->src[slot == 0 ? 1 : 0];
            check(other && source->data && weight->data, "graph consumer missing a bound source");
            const auto base = reinterpret_cast<std::uintptr_t>(weight->data);
            const auto start = reinterpret_cast<std::uintptr_t>(source->data);
            check(start >= base && start - base <= ggml_nbytes(weight) &&
                  ggml_nbytes(source) <= ggml_nbytes(weight) - (start - base), "graph view outside weight storage");
            Json row = {{"name", weight->name}, {"physical_id", ids_.at(weight)}, {"node", node->name},
                {"op", ggml_op_name(node->op)}, {"slot", slot}, {"weight_shape", shape(weight)},
                {"weight_strides", {weight->nb[0], weight->nb[1], weight->nb[2], weight->nb[3]}},
                {"source_offset", start - base}, {"source_shape", shape(source)}, {"input_shape", shape(other)},
                {"output_shape", shape(node)}, {"output_type", ggml_type_name(node->type)}};
            if (node->op == GGML_OP_GET_ROWS) {
                check(other->type == GGML_TYPE_I32 && ggml_is_contiguous(other), "invalid token index input");
                std::vector<std::int32_t> ids(std::size_t(ggml_nelements(other)));
                ggml_backend_tensor_get(other, ids.data(), 0, ids.size() * sizeof(ids[0]));
                row["gather_rows"] = ids;
            }
            consumed_.insert(weight);
            pending_node_ = node;
            pending_.push_back(std::move(row));
        }
    }
    void after(const ggml_tensor *node) {
        if (node != pending_node_) return;
        const auto digest = output_hash(node);
        for (auto &row : pending_) {
            row["output_sha256"] = digest;
            row["after"] = true;
            phases_.back()["entries"].push_back(std::move(row));
        }
        pending_.clear();
        pending_node_ = nullptr;
    }
    void step_end() const {
        if (!ids_.empty()) check(!pending_node_ && consumed_.size() == ids_.size(), "incomplete all-weight graph coverage");
    }
    void finish() const {
        if (ids_.empty()) return;
        std::ofstream out(directory_ / "all-consumers.json");
        out << Json({{"phases", phases_}}).dump(2) << '\n';
        out.close();
        check(bool(out), "cannot write all-weight consumer evidence");
    }
};
