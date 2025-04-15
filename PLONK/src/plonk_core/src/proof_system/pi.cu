#include "pi.cuh"

SyncedMemory as_evals(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n) {
    SyncedMemory pi = repeat_zero(n);
    void* pi_gpu_data = pi.mutable_gpu_data();
    void* public_inputs_data = public_inputs.mutable_gpu_data();
    caffe_gpu_memcpy(public_inputs.size(), public_inputs_data, pi_gpu_data + pi_pos*sizeof(uint64_t)*fr::Limbs);
    return pi;
}

SyncedMemory into_dense_poly(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n, Intt INTT) {
    SyncedMemory field_pi = to_mont(public_inputs);
    SyncedMemory evals_tensor = as_evals(field_pi, pi_pos, n);
    SyncedMemory pi_coeffs = INTT.forward(evals_tensor);
    return pi_coeffs;
}

void into_dense_poly_(SyncedMemory public_inputs, SyncedMemory pi_coeffs, SyncedMemory w_r, uint64_t pi_pos, uint64_t n, Intt INTT, cudaStream_t stream1, cudaStream_t stream2) {
    SyncedMemory field_pi = to_mont(public_inputs);
    uint64_t size = pi_coeffs.size();
    SyncedMemory pi(size);
    void* pi_ = pi.mutable_gpu_data();
    caffe_gpu_memset_async(size, 0, pi_, stream2);
    void* public_inputs_data = public_inputs.mutable_gpu_data();
    caffe_gpu_memcpy(public_inputs.size(), public_inputs_data, pi_ + pi_pos*sizeof(uint64_t)*fr::Limbs);
    cudaDeviceSynchronize();
    INTT.forward_(pi, pi_coeffs, stream2);
    void* w_r_ = w_r.mutable_cpu_data_async(stream1);
}