#include "PLONK/utils/mont/cpu/mont_arithmetic.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace cpu{
  template <typename T>
  static void to_mont_cpu_template(T* self, int64_t numel) {
      int64_t num_ = numel / num_uint64(self[0]);
      for (auto i = 0; i < num_; i++) {
        self[i].to();
      }
  }

  template <typename T>
  static void to_base_cpu_template(T* self, int64_t numel) {
      int64_t num_ = numel / num_uint64(self[0]);
      for (auto i = 0; i < num_; i++) {
        self[i].from();
      }
  }

  template <typename T>
  static void add_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a[0]);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] + in_b[i];
      }
  }

  template <typename T>
  static void sub_template( 
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a[0]);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] - in_b[i];
      }
  }

  template <typename T>
  static void mul_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a[0]);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] * in_b[i];
      }
  }

  template <typename T>
  static void div_template(
      T* in_a,
      T* in_b,
      T* out_c, int64_t numel) {
      int64_t num_ = numel / num_uint64(in_a[0]);
      for (auto i = 0; i < num_; i++) {
        out_c[i] = in_a[i] / in_b[i];
      }
  }

  template <typename T>
  static void exp_template(T* out, int exp, int64_t numel) {
      if (exp == 1) {
        return;
      }
      int64_t num_ = numel / num_uint64(out[0]);
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
      int64_t num_ = numel / num_uint64(out[0]);
      for (auto i = 0; i < num_; i++) {
        out[i] = 1 / out[i];
      }
  }

  template <typename T>
  static void neg_template(T* out, int64_t numel) {
      int64_t num_ = numel / num_uint64(out[0]);
      for (auto i = 0; i < num_; i++) {
        out[i] = -out[i];
      }
  }


  SyncedMemory& to_mont_cpu(SyncedMemory& input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    to_mont_cpu_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory& to_base_cpu(SyncedMemory& input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    to_base_cpu_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory& add_mod_cpu(SyncedMemory& a, SyncedMemory& b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    add_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void add_mod_cpu_(SyncedMemory& self, SyncedMemory& b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    add_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory& sub_mod_cpu(SyncedMemory& a, SyncedMemory& b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    sub_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void sub_mod_cpu_(SyncedMemory& self, SyncedMemory& b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    sub_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory& mul_mod_cpu(SyncedMemory& a, SyncedMemory& b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    mul_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void mul_mod_cpu_(SyncedMemory& self, SyncedMemory& b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    mul_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory& div_mod_cpu(SyncedMemory& a, SyncedMemory& b) {
    SyncedMemory c(a.size());
    void* a_ = a.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    void* c_ = c.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    div_template(static_cast<fr*>(a_), static_cast<fr*>(b_), static_cast<fr*>(c_), numel);
    return c;
  }

  void div_mod_cpu_(SyncedMemory& self, SyncedMemory& b) {
    void* self_ = self.mutable_cpu_data();
    void* b_ = b.mutable_cpu_data();
    int64_t numel = b.size()/sizeof(uint64_t);
    div_template(static_cast<fr*>(self_), static_cast<fr*>(b_), static_cast<fr*>(self_), numel);
  }

  SyncedMemory& exp_mod_cpu(SyncedMemory& input, int exp) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    exp_template(static_cast<fr*>(out), exp, numel);
    return output;
  }

  SyncedMemory& inv_mod_cpu(SyncedMemory& input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    inv_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory& neg_mod_cpu(SyncedMemory& input) {
    SyncedMemory output(input.size());
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memcpy(out, in, input.size());
    int64_t numel = input.size()/sizeof(uint64_t);
    neg_template(static_cast<fr*>(out), numel);
    return output;
  }

  SyncedMemory& pad_poly_cpu(SyncedMemory& input, int64_t N) {
    SyncedMemory output(N * fr_LIMBS * sizeof(uint64_t));
    void* out = output.mutable_cpu_data();
    void* in = input.mutable_cpu_data();
    memset(out, 0, output.size());
    memcpy(out, in, input.size());
    return output;
  }
}//namespace::cpu

