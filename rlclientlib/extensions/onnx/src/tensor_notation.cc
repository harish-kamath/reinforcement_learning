//#pragma once

#include "tensor_notation.h"
#include "err_constants.h"
#include "api_status.h"

#include <memory>
#include <assert.h>

// for base64 decoding
#include <cpprest/http_client.h>

namespace reinforcement_learning { namespace onnx {

// This implements a very restricted parser/serializer for a JSON-like dialect to
// describe input to the ONNX Runtime
//
// The format is:
// <TENSORS> := "{" <TENSOR-LIST> "}"
// <TENSOR-LIST> := <TENSOR> ["," [<TENSOR-LIST>]]
// <TENSOR> := "\"" <INPUT-NAME> "\":\"" <TENSOR-DATA> "\""
// <TENSOR-DATA> := <DIMS-BASE64> ";" <VALUES-BASE64>
// <DIMS-BASE64> := { base64 encoding of int64[] representing dimensions of the tensor }
// <VALUES-BASE64> := { base64 encoding of float[] representing values of the tensor }
//
// Any other JSON concepts are not allowed. The reason for any relation to JSON is
// the current setup for RLClientLib to log the context as JSON.
// Ideally, we would use protobuf definitions from ONNX to represent the IOContext

namespace tokens {
  const char NULL_TERMINATOR = '\0';
  const char ESCAPE = '\\';
  const char DQUOTE = '\"';
  const char SEMICOLON = ';';
  const char COLON = ':';
  const char COMMA = ',';
  const char OPEN_CBRACKET = '{';
  const char CLOSE_CBRACKET = '}';
}

class TensorParserError
{
private:
  const char* ERROR_PREFIX = "Error parsing TensorNotation at position ";

public:
  TensorParserError(size_t line, size_t col) : _message_builder{ERROR_PREFIX} 
  {
    _message_builder << '(' << line << ":" << col << "): ";
  };

  TensorParserError(TensorParserError& rhs) : _message_builder{rhs._message_builder.str()}
  {
  };

  ~TensorParserError() = default;

  template <typename T>
  TensorParserError& operator<<(const T& value)
  {
    _message_builder << value;
    return *this;
  }

  std::string message()
  {
    return _message_builder.str();
  }

private:
  std::stringstream _message_builder;
};

class TensorParser
{
public:
  // warnings/errors?
  TensorParser(const char* tensor_notation, OnnxRtInputContext& input_context);

  bool cached_parse();

  inline bool succeeded()
  {
    return !_has_error;
  }

  inline std::string error_message()
  {
    return _error_message;
  }

private:
  inline void skip_whitespace()
  {
    while (*_reading_head == ' ' 
        || *_reading_head == '\t' 
        || *_reading_head == '\r'
        || *_reading_head == '\n')
    {
      // This is robust to end-of-string without an explicit check
      // because \0 would kick out of the loop above. Technically,
      // the optimizer should take care of it, though. 
      _reading_head++; 
    }
  }

  inline TensorParserError make_error()
  {
    size_t line = 1;
    size_t col = 1;

    std::for_each(_parse_buffer, _reading_head + 1, 
      [&line, &col](const char& c)
      {
        if (c == '\n')
        {
          line++;
          col = 1;
        }
        else
        {
          col++;
        }
      });

    return {line, col};
  }

  inline void read_character(char c)
  {
    if (*_reading_head == c)
    {
      _reading_head++;
    }
    else
    {
      throw (make_error() << "Expecting '" << c << "'; actual '" << *_reading_head << "'.");
    }
  }

  inline bool is_base64(const char c) const
  {
    // I really wish ASCII was a nicer encoding (i.e. alphanumeric = contiguous range) - it 
    // would make base64 much easier to parse without doing a lot of superfluous operations.
    return (c == '+' /* 43 */ 
        || c == '/' /* 47 */ 
        || (c >= '0' /* 48 */ && c <= '9' /* 57 */)
        || (c >= 'A' /* 65 */ && c <= 'Z' /* 90 */)
        || (c >= 'a' /* 97 */  && c <= 'z' /* 122 */)
        || c == '=' /* padding character */);
  }

  inline bytes_t read_base64()
  {
    _token_start = _reading_head;

    // scan BASE64 sequence
    while (is_base64(*_reading_head))
    { 
      _reading_head++;
    }

    // this performs a copy
    std::string base64string(_token_start, _reading_head - _token_start);

    if (base64string.length() % 4 != 0)
    {
      throw (make_error() << "Base64 string \"" << base64string << "\" length is not divisible by 4: " << base64string.length() << ".");
    }

    bytes_t conversion = ::utility::conversions::from_base64(base64string);
    return conversion;
  }

  inline std::string read_tensor_name()
  {
    // Consume \'
    read_character(tokens::DQUOTE);
    
    _token_start = _reading_head;

    // Scan for the name string
    bool in_escape = false;
    while (true)
    {
      switch (*_reading_head)
      {
        case tokens::DQUOTE:
          if (in_escape) break;
          // If not in an escape sequence, terminate scanning for the name
        case tokens::NULL_TERMINATOR:
          goto end_name_scan;
        case tokens::ESCAPE:
          in_escape = !in_escape;
      }

      _reading_head++;
    }
    end_name_scan:

    const char* token_end = _reading_head;

    // Consume \'
    read_character(tokens::DQUOTE);

    return std::string(_token_start, token_end - _token_start);
  }

  inline tensor_data_t read_tensor_data()
  {
    // Consume \'
    read_character(tokens::DQUOTE);

    // TODO: Support reading type information for the tensor (and later map/sequence)
    // See value_t definition in tensor_notation.h
    // <TYPE_INFO> := '<' VALUE_TYPE '>'
    // <VALUE_TYPE> := "float"
    // read_value_type();
    // read_character<';'>();
    
    // Read base_64(int_64t[]) until ';'
    bytes_t dimensions_base64 = read_base64();

    // Consume ';' - also another bit of validation that dimensions were parsed 
    // meaningfully
    read_character(tokens::SEMICOLON);

    // Read base_64(float[]) until '\''
    bytes_t values_base64 = read_base64();

    // Consume \'
    read_character(tokens::DQUOTE);

    // Should we do the base64 parse here?
    return std::make_pair(dimensions_base64, values_base64);
  }

  inline void read_tensor()
  {
    std::string name = read_tensor_name();

    skip_whitespace();

    read_character(tokens::COLON);

    skip_whitespace();

    tensor_data_t value = read_tensor_data();

    _parse_context.push_input(name, value);
  }

  inline void read_tensor_list()
  {
    skip_whitespace();

    read_character(tokens::OPEN_CBRACKET);
    
    skip_whitespace();

    while (*_reading_head != tokens::CLOSE_CBRACKET 
        && *_reading_head != tokens::NULL_TERMINATOR)
    {
      read_tensor();
      skip_whitespace();
      if (*_reading_head == tokens::COMMA)
      {
        read_character(tokens::COMMA);
        skip_whitespace();
      }
    }

    read_character(tokens::CLOSE_CBRACKET);
  }

private:
  const char* const _parse_buffer;
  
  const char* _token_start;
  const char* _reading_head;

  OnnxRtInputContext& _parse_context;

  bool _has_parsed;
  bool _has_error;
  std::string _error_message;
};

TensorParser::TensorParser(const char* tensor_notation, OnnxRtInputContext& input_context) 
: _parse_buffer{tensor_notation}, 
  _reading_head{tensor_notation}, 
  _token_start{tensor_notation}, 
  _parse_context{input_context},
  _has_parsed{false},
  _has_error{false},
  _error_message{""}
{
}

bool TensorParser::cached_parse()
{
  if (_has_parsed)
  {
    return !_has_error;
  }

  try
  {
    if (_parse_buffer && *_parse_buffer != tokens::NULL_TERMINATOR)
    {
      read_tensor_list();
    }
  }
  catch (const TensorParserError& tpe)
  {
    _has_error = true;
  }

  return !_has_error;
}

int read_tensor_notation(const char* tensor_notation, OnnxRtInputContext* input_context, api_status* status)
{
  TensorParser parser(tensor_notation, *input_context);
  if (!parser.cached_parse())
  {
    RETURN_ERROR_LS(nullptr, status, extension_error)
      << "OnnxExtension: Failed to deserialize tensor: "
      << parser.error_message();
  }

  return error_code::success;
}

}}