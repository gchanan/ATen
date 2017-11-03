#pragma once

#include "ATen/Tensor.h"
#include <sstream>
#include <tuple>

namespace at {

std::vector<int64_t> infer_size(IntList a, IntList b);
std::tuple<std::vector<int64_t>, std::vector<int64_t> > inferExpandGeometry(const Tensor &tensor, IntList sizes);

inline void check_defined(std::vector<Tensor> tensors, const std::string &api_name) {
  for (auto& t : tensors) {
    if (!t.defined()) {
      runtime_error("%s(...) called with an undefined Tensor", api_name.c_str());
    }
  }
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(tensor.sizes()));
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand, const std::string &api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace(tensor, to_expand);
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  return std::make_tuple(to_expand1.expand(tensor.sizes()), to_expand2.expand(tensor.sizes()));
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2,
                                                 const std::string &api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace(tensor, to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const std::string &api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = infer_size(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size(expanded_size12, to_expand3.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size), to_expand3.expand(expanded_size));
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3,
                                                          const std::string &api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(sizes));
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes, const std::string &api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

}
