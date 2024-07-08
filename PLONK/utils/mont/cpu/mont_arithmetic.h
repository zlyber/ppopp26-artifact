#pragma once
#include <assert.h>
#include <stdexcept>
#include <limits>
#include "PLONK/utils/mont/cpu/curve_def.h"
#include "caffe/syncedmem.hpp"

using namespace caffe;
namespace cpu{
    SyncedMemory& to_mont_cpu(SyncedMemory& input);
    SyncedMemory& to_base_cpu(SyncedMemory& input);
    SyncedMemory& add_mod_cpu(SyncedMemory& a, SyncedMemory& b);
    void add_mod_cpu_(SyncedMemory& self, SyncedMemory& b);
    SyncedMemory& sub_mod_cpu(SyncedMemory& a, SyncedMemory& b);
    void sub_mod_cpu_(SyncedMemory& self, SyncedMemory& b);
    SyncedMemory& mul_mod_cpu(SyncedMemory& a, SyncedMemory& b);
    void mul_mod_cpu_(SyncedMemory& self, SyncedMemory& b);
    SyncedMemory& div_mod_cpu(SyncedMemory& a, SyncedMemory& b);
    void div_mod_cpu_(SyncedMemory& self, SyncedMemory& b);
    SyncedMemory& exp_mod_cpu(SyncedMemory& input, int exp);
    SyncedMemory& inv_mod_cpu(SyncedMemory& input);
    SyncedMemory& neg_mod_cpu(SyncedMemory& input);
    SyncedMemory& pad_poly_cpu(SyncedMemory& input, int64_t N);
}