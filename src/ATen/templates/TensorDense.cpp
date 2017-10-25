// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList ${Tensor}::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar ${Tensor}::localScalar() {
  AT_ASSERT(isScalar(),"localScalar() called on Tensor with %d dims",sizes().size());
  return Scalar(${to_at_type}(${THTensor}_get1d(${state,}tensor, 0)));
}
void ${Tensor}::assign_(Scalar s) {
  AT_ASSERT(isScalar(),"assign_() called on Tensor with %d dims",sizes().size());
  ${THTensor}_set1d(${state,}tensor, 0,${to_th_type}(s.to${ScalarName}()));
}

Tensor ${Tensor}::expand(IntList sizes) {
  if (sizes.size() < (size_t)dim()) {
    throw std::runtime_error("the number of sizes provided must be greater or equal to the "
                             "number of dimensions in the tensor");
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(this, sizes);


/*std::unique_ptr<Storage> CPUIntType::storage() const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context));
}
std::unique_ptr<Storage> CPUIntType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context,size));
}
std::unique_ptr<Storage> CPUIntType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUIntStorage(context,data,size,deleter));
}*/
  
    //virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const override;

  std::unique_ptr<Storage> storage(new ${Storage}(context, tensor->storage));
  auto r = type().tensor(*storage, (int64_t)tensor->storageOffset, expandedSizes, expandedStrides);
  //auto r = tensor();
  //r.set_(tensor->storage, (int64_t)tensor->storageOffset, expandedSizes, expandedStrides);
  return r;
}