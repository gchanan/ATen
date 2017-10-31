#pragma once

#include "ATen/ATen.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"
#include <functional>
#include <numeric>
#include <sstream>
#include <vector>


namespace at {
namespace native {

/*
[NativeFunction]
name: split
arg: Tensor self
arg: int64_t split_size
arg: int64_t dim=0
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::split
[/NativeFunction]
*/
static inline std::vector<Tensor> split(const Tensor &self, int64_t split_size, int64_t dim=0) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

/*
[NativeFunction]
name: chunk
arg: Tensor self
arg: int64_t chunks
arg: int64_t dim=0
return: TensorList
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::chunk
[/NativeFunction]
*/
static inline std::vector<Tensor> chunk(const Tensor &self, int64_t chunks, int64_t dim=0) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

/*
[NativeFunction]
name: is_same_size
arg: Tensor self
arg: Tensor other
return: bool
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::is_same_size
[/NativeFunction]
*/
static inline bool is_same_size(const Tensor &self, const Tensor &other) {
  return self.sizes().equals(other.sizes());
}

/*
[NativeFunction]
name: permute
arg: Tensor self
arg: IntList dims
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::permute
[/NativeFunction]
*/
static inline Tensor permute(const Tensor & self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    runtime_error("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      runtime_error("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

/*
[NativeFunction]
name: expand
arg: Tensor self
arg: IntList sizes
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::expand
[/NativeFunction]
*/
static inline Tensor expand(const Tensor &self, IntList sizes) {
  if (sizes.size() < (size_t)self.dim()) {
    throw std::runtime_error("the number of sizes provided must be greater or equal to the "
                             "number of dimensions in the tensor");
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, sizes);

  return self.as_strided(expandedSizes, expandedStrides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

static inline std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor &tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() - 1 ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

/*
[NativeFunction]
name: squeeze
arg: Tensor self
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
static inline Tensor squeeze(const Tensor & self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
static inline Tensor squeeze(const Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze_
arg: Tensor self
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
static inline Tensor squeeze_(Tensor self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: squeeze_
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
static inline Tensor squeeze_(Tensor self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: unsqueeze
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::unsqueeze
[/NativeFunction]
*/
static inline Tensor unsqueeze(const Tensor & self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: unsqueeze_
arg: Tensor self
arg: int64_t dim
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::unsqueeze_
[/NativeFunction]
*/
static inline Tensor unsqueeze_(Tensor self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

/*
[NativeFunction]
name: stack
arg: TensorList list
arg: int64_t dim=0
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::stack
[/NativeFunction]
*/
static inline Tensor stack(TensorList list, int64_t dim=0) {
  if (list.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, list[0].dim() + 1);

  std::vector<Tensor> inputs(list.size());
  for (size_t i = 0; i < list.size(); ++i) {
    inputs[i] = list[i].unsqueeze(dim);
  }
  return at::cat(inputs, dim);
}


static inline Tensor maybeSqueeze(const Tensor & tensor, int64_t dim_tensor1, int64_t dim_tensor2) {
  if (dim_tensor1 == 1) {
    return tensor.squeeze(-2);
  } else if (dim_tensor2 == 1) {
    return tensor.squeeze(-1);
  } else {
    return tensor;
  }
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.

[NativeFunction]
name: matmul
arg: Tensor tensor1
arg: Tensor tensor2
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::matmul
[/NativeFunction]
*/
static inline Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    output_size.insert(output_size.end(), size2.end() - 1, size2.end());

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});

    auto output = t1.mm(t2).view(output_size);
    if (dim_tensor2 == 1) {
      output = output.squeeze(-1);
    }
    return output;
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // ensure each tensor size is at least 3-dimensional
    std::vector<int64_t> tensor1_exp_size(std::max<int64_t>(3 - tensor1.dim(), 0), 1);
    tensor1_exp_size.insert(tensor1_exp_size.end(), tensor1.sizes().begin(), tensor1.sizes().end());

    // rhs needs to be a separate case since we can't freely expand 1s on the rhs, but can on lhs
    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(1) : tensor2;
    std::vector<int64_t> tensor2_exp_size(std::max<int64_t>(3 - tensor2.dim(), 0), 1);
    tensor2_exp_size.insert(tensor2_exp_size.end(), tensor2.sizes().begin(), tensor2.sizes().end());

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    IntList batch_tensor1(tensor1_exp_size.data(), tensor1_exp_size.size() - 2);
    IntList batch_tensor2(tensor2_exp_size.data(), tensor2_exp_size.size() - 2);
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), tensor1_exp_size.end() - 2, tensor1_exp_size.end());

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), tensor2_exp_size.end() - 2, tensor2_exp_size.end());

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view(1, expand_batch_product);
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), tensor1_exp_size.end() - 2, tensor1_exp_size.end());
    std::vector<int64_t> tensor2_bmm_view(1, expand_batch_product);
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), tensor2_exp_size.end() - 2, tensor2_exp_size.end());

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    //total_expansion = expand_batch_portion + (tensor1_exp_size[-2], tensor2_exp_size[-1])
    std::vector<int64_t> total_expansion(expand_batch_portion);
    total_expansion.push_back(tensor1_exp_size[*(tensor1_exp_size.end() - 2)]);
    total_expansion.push_back(tensor1_exp_size[*(tensor1_exp_size.end() - 1)]);

    Tensor output = tensor1_expanded.bmm(tensor2_expanded);
    output = maybeSqueeze(output.view(total_expansion), dim_tensor1, dim_tensor2);
    return output;
  }

  std::ostringstream oss;
  oss << "both arguments to matmul need to be at least 1D,  but they are "
      << dim_tensor1 << "D and " << dim_tensor2 << "D";
  throw std::runtime_error(oss.str());
}

}
}
