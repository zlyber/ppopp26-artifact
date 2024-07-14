#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/fixed_base_scalar_mul.cu"
#include "PLONK/src/plonk_core/src/proof_system/widget/curve_addition.cu"

SyncedMemory& as_evals(SyncedMemory& public_inputs, int pi_pos, int n) {
    
    SyncedMemory& pi = repeat_zero(n);
    void* pi_gpu_data = pi.mutable_gpu_data();
    void* public_inputs_data = public_inputs.mutable_gpu_data();
    caffe_gpu_memcpy(public_inputs.size(), public_inputs_data, pi_gpu_data+pi_pos*sizeof(uint64_t));
    return pi;
}

SyncedMemory& into_dense_poly(SyncedMemory& public_inputs, int pi_pos, int n, Intt& INTT) {
    
    SyncedMemory& evals_tensor = as_evals(public_inputs, pi_pos, n);
    void* evals_tensor_gpu_data= evals_tensor.mutable_gpu_data();
    SyncedMemory& pi_coeffs = INTT.forward(evals_tensor);
    return pi_coeffs;
}