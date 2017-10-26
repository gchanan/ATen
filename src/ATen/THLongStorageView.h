#pragma once

#include "TH/TH.h"

namespace at {

// make a fake storage out of a size, pointer pair...
// used as an argument where THSize and THStride are passed into TH
class THLongStorageView {
public:
  static THLongStorageView makeFromSize(ArrayRef<int64_t> ref) {
    return THLongStorageView(ref, true, false, false);
  }
  // noelem_to_empty is to differentiate strides of empty tensors vs scalars.  In ATen, both may have strides [1],
  // but in TH an empty tensor should have stride [], while a scalar should have stride [1].
  static THLongStorageView makeFromStride(ArrayRef<int64_t> ref, bool noelem_to_empty) {
    return THLongStorageView(ref, false, true, noelem_to_empty);
  }
  static THLongStorageView makeFromLength(ArrayRef<int64_t> ref) {
    return THLongStorageView(ref, false, false, false);
  }
  operator THLongStorage*() {
    if (storage.size == 0 && zero_dim_to_null) {
      return nullptr;
    }
    return &storage;
  }
private:
  // zero_dim_to_one converts an empty ArrayRef into [1]
  // zero_dim_to_null converts an empty ArrayRef into a null THLongStorage
  // noelem_to_empty makes an ArrayRef of [0] into an empty THLongStorage
  THLongStorageView(ArrayRef<int64_t> ref, bool zero_dim_to_one, bool zero_dim_to_null, bool noelem_to_empty)
    : zero_dim_to_null(zero_dim_to_null)
  {
    if(zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH
      one = 1;
      storage.data = &one;
      storage.size = 1;
    } else if (noelem_to_empty && ref.size() == 1 && ref[0] == 0) {
      storage.data = (int64_t*)(ref.data());
      storage.size = 0;
    }
    else {
      storage.data = (int64_t*)(ref.data());
      storage.size = ref.size();
    }
    storage.refcount = 0;
    storage.flag = 0;
    storage.allocator = nullptr;
    storage.allocatorContext = nullptr;
  }
  int64_t one;
  THLongStorage storage;
  bool zero_dim_to_null;
};

}
