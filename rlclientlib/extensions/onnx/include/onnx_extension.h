#pragma once

namespace reinforcement_learning { namespace onnx {
  void register_onnx_factory();
}}

// Constants

namespace reinforcement_learning { namespace name {
  //TODO: Explore and expose useful configuration settings here
  const char *const ONNX_PARSE_FEATURE_STRING = "onnx.parse_feature_string";
  const char *const ONNX_OUTPUT_NAME          = "onnx.output_name";
}}

namespace reinforcement_learning { namespace value {
  const char *const ONNXRUNTIME_MODEL = "ONNXRUNTIME";
}}