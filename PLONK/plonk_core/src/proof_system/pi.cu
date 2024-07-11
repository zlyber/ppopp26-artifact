#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/plonk_core/src/proof_system/widget/mod.cu"
#include "PLONK/plonk_core/src/proof_system/widget/fixed_base_scalar_mul.cu"
#include "PLONK/plonk_core/src/proof_system/widget/curve_addition.cu"

SyncedMemory& as_evals(SyncedMemory& public_inputs, int pi_pos, int n) {
    // 创建一个大小为 n 的零张量，数据类型为 fr.LIMBS()，fr.TYPE()
    SyncedMemory& pi = torch::zeros({n}, torch::dtype(fr::TYPE()).device(torch::kCPU).requires_grad(false));
    // 在 pi_pos 位置赋值 public_inputs
    pi.index_put_({pi_pos}, public_inputs);
    return pi;
}

SyncedMemory& into_dense_poly(SyncedMemory& public_inputs, int pi_pos, int n, INTT& INTT) {
    // 调用 as_evals 函数
    SyncedMemory& evals_tensor = as_evals(public_inputs, pi_pos, n);
    // 将 evals_tensor 转移到 CUDA 设备上并进行 INTT 操作
    SyncedMemory& pi_coeffs = INTT(evals_tensor.to(torch::kCUDA));
    return pi_coeffs;
}