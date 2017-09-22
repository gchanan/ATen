#include "ATen/ATen.h"
#include <iostream>
#include <numeric>

using namespace at;

void assert_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  assert(lhs.dim() == rhs.dim());
  assert(lhs.sizes().equals(rhs.sizes()));
}

int main() {
  Type & T = CPU(kFloat);

  std::vector<std::vector<int64_t> > sizes = { {}, {0}, {1, 1}, {2}};

  auto t = T.ones({2});

  // construct a tensor of each size and verify that the dim, sizes, strides, etc.
  // match what was requested.
  for (auto s = sizes.begin(); s != sizes.end(); ++ s) {
    auto t = T.ones(*s);
    assert(t.dim() == s->size());
    assert(t.ndimension() == s->size());
    assert(t.sizes().equals(*s));
    assert(t.strides().size() == s->size());
    auto numel = std::accumulate(s->begin(), s->end(), 1, std::multiplies<int64_t>());
    assert(t.numel() == numel);
    // verify we can output
    std::cout << t << std::endl;

    // set_
    auto t2 = T.ones(*s);
    t2.set_();
    assert_equal_size_dim(t2, T.ones({0}));

    // unsqueeze
    if (t.numel() != 0) {
      if (t.dim() > 0) {
        assert(t.unsqueeze(0).dim() == t.dim() + 1);
      } else {
        // FIXME: should be able to remove this if/else, unsqueezing a scalar should give 1-dimension
        assert(t.unsqueeze(0).dim() == t.dim() + 2);
      }
    } else {
      try {
        t.unsqueeze(0);
        assert (false);
      } catch (std::runtime_error &e) {}
    }

    // squeeze
    if (t.dim() > 0 && t.sizes()[0] == 1) {
      assert(t.squeeze(0).dim() == t.dim() - 1);
    } else if (t.dim() == 0 || t.numel() == 0)  {
      try {
        t.squeeze(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    } else {
      // In PyTorch, it is a no-op to try to squeeze a dimension that has size != 1;
      // in NumPy this is an error.
      assert(t.squeeze(0).dim() == t.dim());
    }

    // reduce
    if (t.dim() > 0 && t.numel() != 0) {
      // FIXME: the max should be 0, but we don't reduce down to scalars properly yet
      assert(t.sum(0).dim() == std::max<int64_t>(t.dim() - 1, 1));
    } else if (t.dim() == 0) {
      try {
        t.sum(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    } else {
      // FIXME: you should be able to reduce over size {0}
      try {
        t.sum(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    }

    if (t.dim() > 0 && t.numel() != 0) {
      assert(t[0].dim() == std::max<int64_t>(t.dim() - 1, 0));
    } else if (t.dim() == 0) {
      try {
        t[0];
        assert(false);
      } catch (std::runtime_error &e) {}
    }
  }

  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    for (auto rhs_it = sizes.begin(); rhs_it != sizes.end(); ++rhs_it) {
      // is_same_size should only match if they are the same shape
      {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          if(*lhs_it != *rhs_it) {
            assert(!lhs.is_same_size(rhs));
            assert(!rhs.is_same_size(lhs));
          }
      }
      // forced size functions (resize_, resize_as, set_)
      {
        // resize_
        {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          lhs.resize_(*rhs_it);
          assert_equal_size_dim(lhs, rhs);
        }
        // resize_as_
        {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          lhs.resize_as_(rhs);
          assert_equal_size_dim(lhs, rhs);
        }
        // set_
        {
          {
            // with tensor
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            lhs.set_(rhs);
            assert_equal_size_dim(lhs, rhs);
          }
          {
            // with storage
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            auto storage = T.storage(rhs.numel());
            lhs.set_(*storage);
            // should not be dim 0 because an empty storage is dim 1; all other storages aren't scalars
            assert(lhs.dim() != 0);
          }
          {
            // with storage
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            auto storage = T.storage(rhs.numel());
            lhs.set_(*storage, rhs.storage_offset(), rhs.sizes(), rhs.strides());
            assert_equal_size_dim(lhs, rhs);
          }
        }
      }

      // view
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        try {
          auto result = lhs.view(rhs_size);
          assert(lhs.numel() == rhs.numel());
          assert_equal_size_dim(result, rhs);
        } catch (std::runtime_error &e) {
          assert(lhs.numel() != rhs.numel());
        }
      }

      // expand
      {
        auto lhs = T.ones(*lhs_it);
        auto lhs_size = *lhs_it;
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        bool should_pass = lhs_size.size() <= rhs_size.size();
        for (auto lhs_dim_it = lhs_size.rbegin(); lhs_dim_it != lhs_size.rend(); ++lhs_dim_it) {
          for (auto rhs_dim_it = rhs_size.rbegin(); rhs_dim_it != rhs_size.rend(); ++rhs_dim_it) {
            if (*lhs_dim_it != 1 && *lhs_dim_it != *rhs_dim_it) {
              should_pass = false;
              break;
            }
          }
        }
        try {
          auto result = lhs.expand(rhs_size);
          assert(should_pass);
          assert_equal_size_dim(result, rhs);
        } catch (std::runtime_error &e) {
          assert(!should_pass);
        }
      }
    }
  }

  return 0;
}
