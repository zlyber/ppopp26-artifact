#pragma once
#include <iostream>
#include "caffe/syncedmem.hpp"
#include "PLONK/utils/mont/cuda/curve_def.cuh"
#include "PLONK/utils/mont/cpu/mont_arithmetic.h"
#include "PLONK/utils/mont/cuda/mont_arithmetic.cuh"
#include "PLONK/utils/zkp/cpu/msmcollect.h"
#include "PLONK/utils/zkp/cuda/zksnark.cuh"
using namespace caffe;


SyncedMemory& to_mont(SyncedMemory& input);

SyncedMemory& to_base(SyncedMemory& input);

SyncedMemory& neg_mod(SyncedMemory& input);

SyncedMemory& inv_mod(SyncedMemory& input);

SyncedMemory& add_mod(SyncedMemory& input1, SyncedMemory& input2);
void add_mod_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& sub_mod(SyncedMemory& input1, SyncedMemory& input2);
void sub_mod_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& mul_mod(SyncedMemory& input1, SyncedMemory& input2);
void mul_mod_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& div_mod(SyncedMemory& input1, SyncedMemory& input2);
void div_mod_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& exp_mod(SyncedMemory& input, int exp);

SyncedMemory& add_mod_scalar(SyncedMemory& input1, SyncedMemory& input2);
void add_mod_scalar_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& sub_mod_scalar(SyncedMemory& input1, SyncedMemory& input2);
void sub_mod_scalar_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& mul_mod_scalar(SyncedMemory& input1, SyncedMemory& input2);
void mul_mod_scalar_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& div_mod_scalar(SyncedMemory& input1, SyncedMemory& input2);
void div_mod_scalar_(SyncedMemory& input1, SyncedMemory& input2);

SyncedMemory& gen_sequence(int N, SyncedMemory& x);

SyncedMemory& repeat_to_poly(SyncedMemory& x, int N);

SyncedMemory& evaluate(SyncedMemory& poly, SyncedMemory& x);

SyncedMemory& poly_div_poly(SyncedMemory& divid, SyncedMemory& c);

SyncedMemory& pad_poly(SyncedMemory& x, int N);

SyncedMemory& repeat_zero(int N);

SyncedMemory& accumulate_mul_poly(SyncedMemory& product);

class Ntt{
public:
    SyncedMemory Params;

    Ntt(int domain_size);

    SyncedMemory& forward(SyncedMemory& input);
};

class Intt {
public:
    SyncedMemory Params;

    Intt(int domain_size);

    SyncedMemory& forward(SyncedMemory& input);
};

class Ntt_coset {
public:
    SyncedMemory Params;
    int Size;

    Ntt_coset(int domain_size, int coset_size);

    SyncedMemory& forward(SyncedMemory& input);
};

class Intt_coset {
public:
    SyncedMemory Params;

    Intt_coset(int domain_size);

    SyncedMemory& forward(SyncedMemory& input);
};

SyncedMemory& multi_scalar_mult(SyncedMemory& points, SyncedMemory& scalars);

bool gt_zkp(SyncedMemory& a, SyncedMemory& b);

SyncedMemory& compress(SyncedMemory& t_0, SyncedMemory& t_1, SyncedMemory& t_2, SyncedMemory& t_3, 
                       SyncedMemory& challenge);

SyncedMemory& compute_query_table(SyncedMemory& q_lookup, 
                        SyncedMemory& w_l_scalar ,SyncedMemory& w_r_scalar , SyncedMemory& w_o_scalar,
                        SyncedMemory& w_4_scalar, SyncedMemory& t_poly, SyncedMemory& challenge);

