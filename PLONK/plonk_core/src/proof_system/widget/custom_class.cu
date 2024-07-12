#include <stdint.h>
#include <vector>
#include <iostream>
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/src/arithmetic.cu"
#include "PLONK/src/bls12_381/edwards.h"
class Custom_class {
public:
    SyncedMemory& a_next_eval;
    SyncedMemory& b_next_eval;
    SyncedMemory& d_next_eval;
    SyncedMemory& q_l_eval;
    SyncedMemory& q_r_eval;
    SyncedMemory& q_c_eval;
    SyncedMemory& q_arith_eval;
    SyncedMemory& q_lookup_eval;
    SyncedMemory& q_hl_eval;
    SyncedMemory& q_hr_eval;
    SyncedMemory& q_h4_eval;

public:
    Custom_class(SyncedMemory& a, SyncedMemory& b, SyncedMemory& d, SyncedMemory& q_l, SyncedMemory& q_r, SyncedMemory& q_c,
                 SyncedMemory& q_arith, SyncedMemory& q_lookup, SyncedMemory& q_hl, SyncedMemory& q_hr, SyncedMemory& q_h4)
        : a_next_eval(a), b_next_eval(b), d_next_eval(d), q_l_eval(q_l), q_r_eval(q_r), q_c_eval(q_c),
          q_arith_eval(q_arith), q_lookup_eval(q_lookup), q_hl_eval(q_hl), q_hr_eval(q_hr), q_h4_eval(q_h4) {}
};
