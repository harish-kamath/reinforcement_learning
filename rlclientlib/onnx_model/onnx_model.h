#pragma once
#include "model_mgmt.h"
//#include "safe_vw.h"
//#include "../utility/versioned_object_pool.h"
#include <core/session/onnxruntime_cxx_api.h>

namespace reinforcement_learning {
  class i_trace;
}

namespace reinforcement_learning { namespace model_management {
  class onnx_model : public i_model {
  public:
    onnx_model(i_trace* trace_logger, const char* app_id, int thread_pool_size);
    int update(const model_data& data, bool& model_ready, api_status* status = nullptr) override;
    int choose_rank(uint64_t rnd_seed, const char* features, std::vector<int>& action_ids, std::vector<float>& action_pdf, std::string& model_version, api_status* status = nullptr) override;
  private:
    i_trace* _trace_logger;

    Ort::Env _env;
    Ort::Allocator _allocator;
    Ort::SessionOptions _session_options;

    std::shared_ptr<Ort::Session> _master_session;
  };
}}
