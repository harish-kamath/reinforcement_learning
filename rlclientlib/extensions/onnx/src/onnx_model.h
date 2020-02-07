#pragma once
#include <string>

#include "model_mgmt.h"

#include <core/session/onnxruntime_cxx_api.h>

namespace reinforcement_learning {
  class i_trace;
}

namespace reinforcement_learning { namespace onnx {
  class onnx_model : public model_management::i_model {
  public:
    onnx_model(onnx_model&&) = delete;
    onnx_model& operator=(onnx_model&&) = delete;

  public:
    onnx_model(i_trace* trace_logger, const char* app_id, const char* output_name, bool parse_feature_string);
    int update(const model_management::model_data& data, bool& model_ready, api_status* status = nullptr) override;
    int choose_rank(uint64_t rnd_seed, const char* features, std::vector<int>& action_ids, std::vector<float>& action_pdf, std::string& model_version, api_status* status = nullptr) override;
  private:
    i_trace* _trace_logger;
    std::string _output_name;
    size_t _output_index;
    const bool _parse_feature_string;

    Ort::Env _env;
    Ort::AllocatorWithDefaultOptions _allocator;
    Ort::SessionOptions _session_options;

    std::shared_ptr<Ort::Session> _master_session;
  };
}}
