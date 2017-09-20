#include "ATen/ATen.h"
#include <iostream>
#include <numeric>

using namespace at;

void assert_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  
  //std::cout << lhs.dim() << " " << rhs.dim() << std::endl;
  assert(lhs.dim() == rhs.dim());
  assert(lhs.sizes().equals(rhs.sizes()));
}

int main() {
  Type & T = CPU(kFloat);

  std::vector<std::vector<int64_t> > sizes = { {}, {0}, {1, 1}, {2}};

  // construct a tensor of each size and verify that the dim, sizes, strides, etc.
  // match what you requested.
  for (auto s = sizes.begin(); s != sizes.end(); ++ s) {
    auto t = T.ones(*s);
    std::cout << "case " << *s << std::endl;
    t.sizes()[0] = 5;
    assert(t.dim() == s->size());
    assert(t.sizes().equals(*s));
    assert(t.strides().size() == s->size());
    auto multi = std::accumulate(s->begin(), s->end(), 1, std::multiplies<int64_t>());
    assert(t.numel() == multi);

    // squeeze/unsqueeze
    //t.unsqueeze();
    //t.unsqueeze(0);
    //t.squeeze(0);
    //t.unsqueeze(0);
    
    // reduce
    // t.sum(0);
    
    // accessor()
    // accessor via []
  }

  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    for (auto rhs_it = sizes.begin(); rhs_it != sizes.end(); ++rhs_it) {
      // is_same_size should not match
      {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          assert(!lhs.is_same_size(rhs));
          assert(!rhs.is_same_size(lhs));
      }
      // resize_as_
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        lhs.resize_as_(rhs);
        //std::cout << "checking " << *lhs_it << " " << *rhs_it << std::endl;
        //assert_equal_size_dim(lhs, rhs);
        if(lhs.dim() != rhs.dim() || !lhs.sizes().equals(rhs.sizes())) {
          std::cout << "failed resize as " << *lhs_it << " " << *rhs_it << std::endl;
        }
      }

      // view
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        //std::cout << "testing" << lhs.dim() << " " << rhs.dim() << " " << lhs.numel() << " " << rhs.numel() << std::endl;
        
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
        //std::cout << "testing" << lhs.dim() << " " << rhs.dim() << " " << lhs.numel() << " " << rhs.numel() << std::endl;
        bool should_pass = true;
        for (auto lhs_dim_it = lhs_size.rbegin(); lhs_dim_it != lhs_size.rend(); ++lhs_dim_it) {
          for (auto rhs_dim_it = rhs_size.rbegin(); rhs_dim_it != rhs_size.rend(); ++rhs_dim_it) {
            if (*lhs_dim_it != 1 && *rhs_dim_it != 1 && *lhs_dim_it != *rhs_dim_it) {
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
          if (!should_pass) std::cout << "failed expand "<< lhs_size << " " << rhs_size << std::endl;
          //assert(!should_pass);
        }
        // to pass:
        // either the dimensions must match
        // the dimension must 
      }
    }
  }

  return 0;
}
