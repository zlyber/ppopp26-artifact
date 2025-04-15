#include "mont_arithmetic.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace cpu{
  template <typename T>
  static void to_mont_cpu_template(T* self, int64_t numel) {
      int64_t num_ = numel / num_uint64(self);
      for (auto i = 0; i < num_; i++) {
        self[i].to();
      }
  }

  template <typename T>
  static void to_base_cpu_template(T* self, int64_t numel) {
      int64_t num_ = numel / num_uint64(self);
      for (auto i = 0; i < num_; i++) {
        self[i].from();
      }
  }

  template <typename T>
  static void add_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] + in_b[i];
      }
  }

  template <typename T>
  static void sub_template( 
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] - in_b[i];
      }
  }

  template <typename T>
  static void mul_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] * in_b[i];
      }
  }

  template <typename T>
  static void div_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] / in_b[i];
      }
  }

  template <typename T>
  static void exp_template(T* out, int exp, int64_t numel) {
      if (exp == 1) {
        return;
      }
      int64_t num_ = numel / num_uint64(out);
      if (exp == 0) {
        for (auto i = 0; i < num_; i++) {
          out[i] = out[i].one();
        }
      } else {
        for (auto i = 0; i < num_; i++) {
          out[i] = out[i] ^ exp;
        }
      }
  }

  template <typename T>
  static void inv_template(T* out, int64_t numel) {
      int64_t num_ = numel / num_uint64(out);
      for (auto i = 0; i < num_; i++) {
        out[i] = 1 / out[i];
      }
  }

  template <typename T>
  static void neg_template(T* out, int64_t numel) {
      int64_t num_ = numel / num_uint64(out);
      for (auto i = 0; i < num_; i++) {
        out[i] = -out[i];
      }
  }


  SyncedMemory to_mont_cpu(SyncedMemory input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    if(!input.type_flag){
      to_mont_cpu_template(static_cast<fq*>(out), numel);
      return output;
    }
    to_mont_cpu_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory to_base_cpu(SyncedMemory input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    if(!input.type_flag){
      to_base_cpu_template(static_cast<fq*>(out), numel);
      return output;
    }
    to_base_cpu_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory add_mod_cpu(SyncedMemory a, SyncedMemory b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!a.type_flag)
      {
        c.type_flag = false;
        add_template(static_cast<fq*>(a_), static_cast<fq*>(b_), static_cast<fq*>(c_), numel);
        return c;
      }
    add_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void add_mod_cpu_(SyncedMemory self, SyncedMemory b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!self.type_flag)
      {
        add_template(static_cast<fq*>(self_), static_cast<fq*>(b_), static_cast<fq*>(self_), numel);
        return;
      }
    add_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory sub_mod_cpu(SyncedMemory a, SyncedMemory b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!a.type_flag)
      {
        c.type_flag = false;
        sub_template(static_cast<fq*>(a_), static_cast<fq*>(b_), static_cast<fq*>(c_), numel);
        return c;
      }
    sub_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void sub_mod_cpu_(SyncedMemory self, SyncedMemory b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!self.type_flag)
      {
        sub_template(static_cast<fq*>(self_), static_cast<fq*>(b_), static_cast<fq*>(self_), numel);
        return;
      }
    sub_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory mul_mod_cpu(SyncedMemory a, SyncedMemory b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!a.type_flag)
      {
        c.type_flag = false;
        mul_template(static_cast<fq*>(a_), static_cast<fq*>(b_), static_cast<fq*>(c_), numel);
        return c;
      }
    mul_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void mul_mod_cpu_(SyncedMemory self, SyncedMemory b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!self.type_flag)
      {
        mul_template(static_cast<fq*>(self_), static_cast<fq*>(b_), static_cast<fq*>(self_), numel);
        return;
      }
    mul_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory div_mod_cpu(SyncedMemory a, SyncedMemory b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!a.type_flag)
      {
        c.type_flag = false;
        div_template(static_cast<fq*>(a_), static_cast<fq*>(b_), static_cast<fq*>(c_), numel);
        return c;
      }
    div_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void div_mod_cpu_(SyncedMemory self, SyncedMemory b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    if(!self.type_flag)
      {
        div_template(static_cast<fq*>(self_), static_cast<fq*>(b_), static_cast<fq*>(self_), numel);
        return;
      }
    div_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory exp_mod_cpu(SyncedMemory input, int exp) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    exp_template(static_cast<fr*>(out), exp, numel);
    return output;
  }

  SyncedMemory inv_mod_cpu(SyncedMemory input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    inv_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory neg_mod_cpu(SyncedMemory input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    if(!input.type_flag)
      {
        output.type_flag = false;
        neg_template(static_cast<fq*>(out), numel);
        return output;
      }
    neg_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory pad_poly_cpu(SyncedMemory input, int64_t N) {
    SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t));
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memset(out, 0, output.size());
    memcpy(out, in, input.size());
    return output;
  }

  void pad_poly_cpu_(SyncedMemory input, SyncedMemory output, int64_t N) {
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memset(out, 0, output.size());
    memcpy(out, in, input.size());
  }

  // Function to reverse bits for an unsigned integer type
  uint32_t bit_rev_32(uint32_t i, uint32_t nbits) {
      uint32_t rev = 0;
      for (uint32_t j = 0; j < nbits; ++j) {
          rev = (rev << 1) | (i & 1);
          i >>= 1;
      }
      return rev;
  }

  // Overload for long long integer (64-bit)
  uint64_t bit_rev_64(uint64_t i, uint32_t nbits) {
      uint64_t rev = 0;
      for (uint32_t j = 0; j < nbits; ++j) {
          rev = (rev << 1) | (i & 1);
          i >>= 1;
      }
      return rev;
  }

  uint32_t bit_rev(uint32_t i, uint32_t nbits) {
      if (sizeof(i) == 4 || nbits <= 32) {
          return bit_rev_32((i), nbits);
      } else {
          return bit_rev_64((i), nbits);
      }
  }

  template <typename fr>
  void bit_rev_permutation(
      fr* d_out,
      const fr* d_in,
      int lg_N) {
    uint64_t N = 1 << lg_N;
    for(int i = 0; i< N; i++){
      uint32_t r = bit_rev(i, lg_N);
      if (i < r || (d_out != d_in && i == r)) {
          fr t0 = d_in[i];
          fr t1 = d_in[r];
          d_out[r] = t0;
          d_out[i] = t1;
      }
    }
  }


  // Worker function for each thread
  template <typename fr>
  void bit_rev_permutation_worker(
      fr* d_out,
      const fr* d_in,
      int lg_N,
      uint64_t start_idx,
      uint64_t end_idx) {

      uint64_t N = 1 << lg_N;
      for (uint64_t i = start_idx; i < end_idx; ++i) {
          uint32_t r = bit_rev(i, lg_N);
          // std::lock_guard<std::mutex> lock(mtx);
          if (i < r || (d_out != d_in && i == r)) {
              fr t0 = d_in[i];
              fr t1 = d_in[r];
              d_out[r] = t0;
              d_out[i] = t1;
          }
      }
  }

  // Multithreaded version of bit_rev_permutation
  template <typename fr>
  void bit_rev_permutation_parallel(
      fr* d_out,
      const fr* d_in,
      int lg_N) {

      uint64_t N = 1 << lg_N;
      uint32_t num_threads = std::thread::hardware_concurrency();  // 获取硬件并发线程数
      num_threads = 1 << (static_cast<int>(std::log2(num_threads)));
      uint32_t chunk_size = N / num_threads;

      std::vector<std::thread> threads;

      // 启动多个线程来并行执行比特反转置换
      for (uint32_t t = 0; t < num_threads; ++t) {
          uint64_t start_idx = t * chunk_size;
          uint64_t end_idx = (t == num_threads - 1) ? N : start_idx + chunk_size;
          threads.push_back(std::thread(bit_rev_permutation_worker<fr>, d_out, d_in, lg_N, start_idx, end_idx));
      }

      // 等待所有线程完成
      for (auto& t : threads) {
          t.join();
      }
  }

}//namespace::cpu

