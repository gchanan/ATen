#include "ATen/${Tensor}.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"
#include "ATen/ExpandUtils.h"

namespace at {

${Tensor}::${Tensor}(Context* context)
: ${Tensor}(context,${THTensor}_new(${state})) {}

${Tensor}::${Tensor}(Context* context, ${THTensor} * tensor)
: TensorImpl(&context->getType(Backend::${Backend},ScalarType::${ScalarName})),
  tensor(tensor),
  context(context) {}
${Tensor}::~${Tensor}() {
  ${THTensor}_free(${state,} tensor);
}

const char * ${Tensor}::toString() const {
  return "${Tensor}";
}

IntList ${Tensor}::sizes() {
  int64_t d = ${THTensor_nDimension};
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t ${Tensor}::dim() {
  if(isScalar())
    return 0;
  int64_t d = ${THTensor_nDimension};
  // See Note [Undefined-dim versus 0-dim]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * ${Tensor}::typeString() {
  return "${Type}";
}
void * ${Tensor}::unsafeGetTH(bool retain) {
  if (retain)
      ${THTensor}_retain(${state,} tensor);
  return tensor;
}

Tensor ${Tensor}::expand(IntList sizes) {
  if (sizes.size() < (size_t)dim()) {
    throw std::runtime_error("the number of sizes provided must be greater or equal to the "
                             "number of dimensions in the tensor");
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(this, sizes);

  at::Tensor r;
  r.set_(tensor->storage, (int64_t)tensor->storageOffset, expandedSizes, expandedStrides);
  return r;
}

${TensorDenseOrSparse}

}
