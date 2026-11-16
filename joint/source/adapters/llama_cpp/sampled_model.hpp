#pragma once
#include "ggml.h"
bool sampled_model_active();
void sampled_model_configure(const char *,const char *);
void sampled_model_bind(const ggml_tensor *const *,unsigned);
void sampled_model_step_begin(int);
void sampled_model_graph_node(const ggml_tensor *);
void sampled_model_step_end();
void sampled_model_finish();
void sampled_model_abort();
