#include "ATen/ExpandUtils.h"

namespace at {

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferExpandGeometry(const TensorImpl *tensor, IntList sizes) {
  //throw yourmamaexception();
  /*
  int64_t *tensorSizes, int64_t *tensorStrides, int64_t tensorDim,
                                        THLongStorage *sizes, int64_t **expandedSizes, int64_t **expandedStrides) {
  ptrdiff_t ndim = THLongStorage_size(sizes);

  int64_t *expandedSizesCalc = THAlloc(sizeof(int64_t)*ndim);
  int64_t *expandedStridesCalc = THAlloc(sizeof(int64_t)*ndim);

  // create a new geometry for the tensors
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensorDim - 1 - offset;
    int64_t size = (dim >= 0) ? tensorSizes[dim] : 1;
    int64_t stride = (dim >= 0) ?
        tensorStrides[dim] : expandedSizesCalc[i + 1] * expandedStridesCalc[i+1];
    int64_t targetSize = THLongStorage_data(sizes)[i];
    if (targetSize == -1) {
      if (dim < 0) {
        THFree(expandedSizesCalc);
        THFree(expandedStridesCalc);
        snprintf(error_buffer, buffer_len, "The expanded size of the tensor (%" PRId64 ") isn't allowed in a leading, non-existing dimension %" PRId64 ".", targetSize, i);
        return -1;
      } else {
        targetSize = size;
      }
    }
    if (size != targetSize) {
      if (size == 1) {
        size = targetSize;
        stride = 0;
      } else {
        THFree(expandedSizesCalc);
        THFree(expandedStridesCalc);
        snprintf(error_buffer, buffer_len, "The expanded size of the tensor (%" PRId64 ") must match the existing size (%" PRId64 ") at "
                 "non-singleton dimension %" PRId64 ".", targetSize, size, i);
        return -1;
      }
    }
    expandedSizesCalc[i] = size;
    expandedStridesCalc[i] = stride;
  }
  *expandedSizes = expandedSizesCalc;
  *expandedStrides = expandedStridesCalc;
  return 0;*/
  std::vector<int64_t> sizes2(0);
  std::vector<int64_t> strides2(0);
  return std::tuple<std::vector<int64_t>, std::vector<int64_t>>(sizes2, strides2);
}

Tensor expand(const Tensor &tensor, IntList sizes) {
  if (sizes.size() < (size_t)tensor.dim()) {
    throw std::runtime_error("the number of sizes provided must be greater or equal to the "
                             "number of dimensions in the tensor");
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(tensor.get(), sizes);

  at::Tensor r;
  //at::Storage storage;
  //r.set_(storage, r.storage_offset(), expandedSizes, expandedStrides);
  return r;
  //THTensor_(setStorageNd)(r, THTensor_(storage)(tensor), THTensor_(storageOffset)(tensor),
  //                        THLongStorage_size(sizes), expandedSizes, expandedStrides);
}

std::vector<int64_t> infer_size2(IntList a, IntList b) {
  auto dimsA = a.size();
  auto dimsB = b.size();
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes(ndim);

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? a[dimA] : 1;
    long sizeB = (dimB >= 0) ? b[dimB] : 1;
    if (sizeA == sizeB || sizeA == 1 || sizeB == 1) {
      expandedSizes[i] = std::max(sizeA, sizeB);
    } else {
      std::ostringstream oss;
      oss << "The size of tensor a (" << sizeA << ") must match the size of tensor b ("
          << sizeB << ") at non-singleton dimension " << i;
      throw std::runtime_error(oss.str());
    }
  }

  return expandedSizes;
}

}
