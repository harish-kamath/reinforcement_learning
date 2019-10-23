#include <memory>
#include <sstream>
#include <assert.h>

#include "tensor_notation.h"

#include "trace_logger.h"
#include "err_constants.h"
#include "onnx_model.h"
#include "str_util.h"
#include "api_status.h"

#include "factory_resolver.h"

// for base64 decoding
#include <cpprest/http_client.h>

namespace reinforcement_learning { namespace onnx {

  inline void OrtLogCallback(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message)
  {
    i_trace* trace_logger = static_cast<i_trace*>(param);

    int loglevel = LEVEL_ERROR;
    switch (severity)
    {
      case ORT_LOGGING_LEVEL_VERBOSE:
        loglevel = LEVEL_DEBUG;
      break;
      
      case ORT_LOGGING_LEVEL_INFO:
        loglevel = LEVEL_INFO;
      break;

      case ORT_LOGGING_LEVEL_WARNING:
        loglevel = LEVEL_WARN;
      break;

      case ORT_LOGGING_LEVEL_FATAL: // TODO: Should this be a background error?
      case ORT_LOGGING_LEVEL_ERROR:
        loglevel = LEVEL_ERROR;
      break;
    }

    std::stringstream buf;
    buf << "[onnxruntime, modelid=" << logid << "]: " << message;

    TRACE_LOG(trace_logger, loglevel, buf.str());
  }

  onnx_model::onnx_model(i_trace* trace_logger, const char* app_id, const char* output_name, int thread_pool_size, bool parse_feature_string) :
    _trace_logger(trace_logger),
    _output_name(output_name),
    _parse_feature_string(parse_feature_string),
    _env(Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, app_id, OrtLogCallback, trace_logger)),
    _allocator(Ort::Allocator::CreateDefault())
  {
    _session_options.SetThreadPoolSize(thread_pool_size);

    // 0 -> To disable all optimizations
    // 1 -> To enable basic optimizations (Such as redundant node removals)
    // 2 -> To enable all optimizations (Includes level 1 + more complex optimizations like node fusions)
    _session_options.SetGraphOptimizationLevel(2); // Make it FAST
  }

  int onnx_model::update(const model_management::model_data& data, bool& model_ready, api_status* status) {
    try {
      TRACE_INFO(_trace_logger, utility::concat("Received new model data. With size ", data.data_sz()));
	  
      Ort::Session* new_session = nullptr;
      if (data.data_sz() <= 0)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Empty model data.";
      }

      new_session = new Ort::Session(_env, data.data(), data.data_sz(), _session_options);
      
      // Validate that the model makes sense
      // Rules: 
      // 1. There are N inputs, which are all tensors of floats
      // 2. There is an output with the provided name, which is a tensor of floats

      size_t input_count = new_session->GetInputCount();
      for (int i = 0; i < input_count; i++)
      {
        // TODO: Support more input types (by making the input interface richer)
        Ort::TypeInfo input_type_info = new_session->GetInputTypeInfo(i);
        if (input_type_info.GetONNXType() != ONNX_TYPE_TENSOR ||
            input_type_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
          RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid input type. Expected: tensor<float>.";
        }
      }

      bool found_output = false;
      size_t output_index = 0;
      size_t output_count = new_session->GetOutputCount();
      for (output_index = 0; output_index < output_count; output_index++)
      {
        char* output_name = new_session->GetOutputName(output_index, _allocator);
        
        if (_output_name == output_name)
        {
          found_output = true;
        }

        _allocator.Free(output_name);

        if (found_output)
        {
          break;
        }
      }

      if (!found_output)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Could not find output with name '" << _output_name << "' in model.";
      }
      
      // TODO: Support more output types
      Ort::TypeInfo output_type_info = new_session->GetOutputTypeInfo(output_index);
      if (output_type_info.GetONNXType() != ONNX_TYPE_TENSOR ||
          output_type_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid output type. Expected: tensor<float>.";
      }

      // TODO: Should we add additional checks to make sure the next two sets are atomic?
      _output_index = output_index;

      // TODO: Should this be moved/forwarded?
      _master_session = std::shared_ptr<Ort::Session>(new_session);
    }
    catch(const std::exception& e) {
      RETURN_ERROR_LS(_trace_logger, status, model_update_error) << e.what();
    }
    catch ( ... ) {
      RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Unknown error";
    }
    
    model_ready = true;
    return error_code::success;
  }

  int onnx_model::choose_rank(uint64_t rnd_seed, 
    const char* features, 
    std::vector<int>& action_ids, 
    std::vector<float>& action_pdf, 
    std::string& model_version,
    api_status* status)
  {
    std::shared_ptr<Ort::Session> local_session = _master_session;
    if (!(bool(local_session)))
    {
      // Model is not ready
      RETURN_ERROR_LS(_trace_logger, status, model_rank_error) << "No model loaded.";
    }
    
    // TODO: Support GPU scoring
    Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    OnnxRtInputContext input_context(allocator_info);
    if (_parse_feature_string)
    {
      RETURN_IF_FAIL(read_tensor_notation(features, &input_context, status));
    }
    else
    {
      // TODO: This is a hook for testing example_builder APIs; for now it is
      // an error to use this path
      RETURN_ERROR_LS(_trace_logger, status, model_rank_error) << "Using parse_feature_string=false not implemented. See onnx_model.cc.";
    }
    
    Ort::RunOptions run_options{nullptr};

    std::vector<const char*> input_names = input_context.input_names();
    std::vector<Ort::Value> inputs = input_context.inputs();
    if (inputs.size() != input_context.input_count())
    {
      // TODO: propagate errors to be more specific about which input(s) is/are bad
      RETURN_ERROR_LS(_trace_logger, status, model_rank_error) << "Could not interpret input values to match expected inputs.";
    }

    auto outputs = local_session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), input_context.input_count(), (const char* const*)&_output_name, 1);
    assert(outputs.size() > _output_index && outputs[_output_index].IsTensor());

    // We cannot be const-correct here, because the only way to read data from the tensor
    // is to query the mutable data field.
    Ort::Value& target_output = outputs[_output_index];

    size_t num_elements = target_output.GetTensorTypeAndShapeInfo().GetElementCount();
    float* floatarr = target_output.GetTensorMutableData<float>();

    for (size_t i = 0; i < num_elements; i++)
    {
      action_ids.push_back(i);
      action_pdf.push_back(floatarr[i]);
    }

    return error_code::success;
  }
}}