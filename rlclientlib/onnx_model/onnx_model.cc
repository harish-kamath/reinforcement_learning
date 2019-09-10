#include <sstream>
#include <assert.h>

#include "trace_logger.h"
#include "err_constants.h"
#include "onnx_model.h"
#include "str_util.h"
#include "api_status.h"

// for base64 decoding
#include <cpprest/http_client.h>

namespace reinforcement_learning { namespace model_management {

  inline void OrtLogCallback(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message)
  {
    i_trace* trace_logger = (i_trace*)param;

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

  onnx_model::onnx_model(i_trace* trace_logger, const char* app_id, int thread_pool_size) :
    _allocator(Ort::Allocator::CreateDefault()), 
    _trace_logger(trace_logger),
    _env(Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, app_id, OrtLogCallback, trace_logger))
  {
    _session_options.SetThreadPoolSize(thread_pool_size);

    // 0 -> To disable all optimizations
    // 1 -> To enable basic optimizations (Such as redundant node removals)
    // 2 -> To enable all optimizations (Includes level 1 + more complex optimizations like node fusions)
    _session_options.SetGraphOptimizationLevel(2); // Make it FAST
  }

  int onnx_model::update(const model_data& data, bool& model_ready, api_status* status) {
    try {
      TRACE_INFO(_trace_logger, utility::concat("Received new model data. With size ", data.data_sz()));
	  
      if (data.data_sz() > 0)
      {
        // safe_vw_factory will create a copy of the model data to use for vw object construction.
        _master_session = std::make_shared<Ort::Session>(_env, data.data(), data.data_sz(), _session_options);
      }

      // Validate that the model makes sense
      // Rules: 
      // 1. There is only a single input, which is a tensor of floats
      // 2. There is only a single output, which is a tensor of floats

      // TODO: Validate input name matches input options instead of requiring it to be
      // exactly one input.
      size_t input_count = _master_session->GetInputCount();
      if (input_count != 1)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid number of inputs. Expected: 1. Actual: " << input_count;
      }

      // TODO: Support more input types (by making the input interface richer)
      Ort::TypeInfo input_type_info = _master_session->GetInputTypeInfo(0);
      if (input_type_info.GetONNXType() != ONNX_TYPE_TENSOR ||
          input_type_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid input type. Expected: tensor<float>.";
      }

      // TODO: Validate output name matches output options instead of requiring it be
      // exactly one output. 
      size_t output_count = _master_session->GetOutputCount();
      if (output_count != 1)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid number of outputs. Expected: 1. Actual: " << output_count;
      }

      // TODO: Support more output types
      Ort::TypeInfo output_type_info = _master_session->GetOutputTypeInfo(0);
      if (output_type_info.GetONNXType() != ONNX_TYPE_TENSOR ||
          output_type_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      {
        RETURN_ERROR_LS(_trace_logger, status, model_update_error) << "Invalid output type. Expected: tensor<float>.";
      }

      model_ready = true;
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
    }
    
    // TODO: Support GPU scoring
    Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // TODO: We should grab these when we update the model
    const char* input_name = local_session->GetInputName(0, _allocator);
    const char* output_name = local_session->GetOutputName(0, _allocator);
    Ort::TypeInfo input_type_info = local_session->GetInputTypeInfo(0);
    Ort::TypeInfo output_type_info = local_session->GetOutputTypeInfo(0);
    Ort::TensorTypeAndShapeInfo input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_tensor_shape = input_tensor_info.GetShape();

    Ort::Value input_tensor(nullptr);
    bool parsed_tensor = false;

    const bool _use_vw_parser = false;
    if (_use_vw_parser)
    {
      throw std::runtime_error("VW Input is not implemented for ONNX");

      // VW Input only supports a very specific interpretation as an input:
      // [256, size(dense_features[b=known-b])]

      // TODO: Recosider this implementation after SparseTensor is available
    }
    else
    {
      // Treat input as Base64 encoded string.
      std::vector<unsigned char> raw_bytes = utility::conversions::from_base64(std::string(features));
      //"{ N1: { _base64: \"{}\" }"

      const size_t expected_byte_count = input_tensor_info.GetElementCount() * sizeof(float);
      if (expected_byte_count != raw_bytes.size())
      {
        RETURN_ERROR_LS(_trace_logger, status, error_code::bad_context_size) << "Expected: " << expected_byte_count << ". Actual: " << raw_bytes.size();
      }

      size_t input_element_count = raw_bytes.size() / sizeof(float);
      float* input_elements = (float*)raw_bytes.data();
      
      input_tensor = Ort::Value::CreateTensor(allocator_info, input_elements, raw_bytes.size(), input_tensor_shape.data(), input_tensor_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      assert(input_tensor.IsTensor());
      parsed_tensor = true;
    }

    if (!parsed_tensor)
    {
      //TODO:
      return RETURN_ERROR_LS(_trace_logger, api_status, error_code::invalid_argument) << "Input could not be reshaped to a float tensor with shape: ";
    }

    auto output_tensors = local_session.Run(Ort::RunOptions{nullptr}, (const char* const*)&input_name, &input_tensor, 1, (const char* const*)&output_name), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    size_t num_elements = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    for (int i = 0; i < num_elements; i++)
    {
      action_ids.push_back(i);
      action_pdf.push_back(floatarr[i]);
    }

    return error_code::success;
  }
}}