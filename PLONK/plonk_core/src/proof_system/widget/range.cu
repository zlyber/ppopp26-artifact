
#include <cuda_runtime.h>
#include "PLONK/plonk_core/src/permutation/constants.cu"
#include "PLONK/utils/function.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/domain.cu"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include <string>
SyncedMemory& _constraints(SyncedMemory& separation_challenge, const WitnessValues& wit_vals, std::map<std::string, SyncedMemory&> custom_vals) {

    SyncedMemory& four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();
   
    SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge);
    SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);

   
    SyncedMemory& b_1_1 = mul_mod_scalar(wit_vals.d_val, four);
    SyncedMemory& f_b1 = sub_mod(wit_vals.c_val, b_1_1);
    SyncedMemory& b_1 = delta(f_b1);

    
    SyncedMemory& b_2_1 = mul_mod(four, wit_vals.c_val);
    SyncedMemory& b_2_2 = sub_mod(wit_vals.b_val, b_2_1);
    SyncedMemory& f_b2 = delta(b_2_2);
    SyncedMemory& b_2 = mul_mod(f_b2, kappa);

    
    SyncedMemory& b_3_1 = mul_mod(four, wit_vals.b_val);
    SyncedMemory& b_3_2 = sub_mod(wit_vals.a_val, b_3_1);
    SyncedMemory& f_b3 = delta(b_3_2);
    SyncedMemory& b_3 = mul_mod(f_b3, kappa_sq);

   
    SyncedMemory& b_4_1 = mul_mod(four, wit_vals.a_val);
    SyncedMemory& b_4_2 = sub_mod(custom_vals["d_next_eval"], b_4_1);
    SyncedMemory& f_b4 = delta(b_4_2);
    SyncedMemory& b_4 = mul_mod(f_b4, kappa_cu);

    
    SyncedMemory& mid1 = add_mod(b_1, b_2);
    SyncedMemory& mid2 = add_mod(mid1, b_3);
    SyncedMemory& mid3 = add_mod(mid2, b_4);
    SyncedMemory& res = mul_mod(mid3, separation_challenge);

    return res;
}

SyncedMemory& quotient_term(SyncedMemory& selector, SyncedMemory& separation_challenge, const WitnessValues& wit_vals, const CustomValues& custom_vals) {
 
    SyncedMemory& four = fr::make_tensor(4);
    void* four_gpu_data=four.mutable_gpu_data();

    
    SyncedMemory& kappa = mul_mod(separation_challenge, separation_challenge); 
    SyncedMemory& kappa_sq = mul_mod(kappa, kappa);
    SyncedMemory& kappa_cu = mul_mod(kappa_sq, kappa);


    SyncedMemory& mid = mul_mod_scalar(wit_vals.d_val, four);
    mid = sub_mod(wit_vals.c_val, mid);
    SyncedMemory& b_1 = delta(mid);

    
    mid = mul_mod_scalar(wit_vals.c_val, four);
    mid = sub_mod(wit_vals.b_val, mid);
    mid = delta(mid);
    void* kappa_gpu_data=kappa.mutable_gpu_data();
    SyncedMemory& b_2 = mul_mod_scalar(mid, kappa);

    SyncedMemory& mid1 = add_mod(b_1, b_2);
    b_1.reset(); b_2.reset(); 

  
    mid = mul_mod_scalar(wit_vals.b_val, four);
    mid = sub_mod(wit_vals.a_val, mid);
    mid = delta(mid);
    void* kappa_sq_gpu_data=kappa_sq.mutable_gpu_data();
    SyncedMemory& b_3 = mul_mod_scalar(mid, kappa_sq);

    mid1 = add_mod(mid1, b_3);
    b_3.reset(); 


    mid = mul_mod_scalar(wit_vals.a_val, four);
    mid = sub_mod(custom_vals["d_next_eval"], mid);
    mid = delta(mid);
    SyncedMemory& b_4 = mul_mod_scalar(mid, kappa_cu.to("cuda"));

    mid = add_mod(mid1, b_4);
    b_4.reset(); 
    
    SyncedMemory& temp = mul_mod_scalar(mid, separation_challenge.to("cuda"));

    SyncedMemory& res = mul_mod(selector, temp);
    return res;
}

SyncedMemory& linearisation_term(SyncedMemory& selector_poly, SyncedMemory& separation_challenge, const WitnessValues& wit_vals, const CustomValues& custom_vals) {
    SyncedMemory& temp = _constraints(separation_challenge, wit_vals, custom_vals);
    SyncedMemory& res;
    if (selector_poly.size(0) == 0) {
        res = selector_poly.clone();
    } else {
        res = mul_mod_scalar(selector_poly, temp);
    }
    return res;
}