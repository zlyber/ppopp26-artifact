#pragma once
#include "../../../bls12_381/fr.cuh"
#include "../../../../utils/function.cuh"

SyncedMemory into_dense_poly(SyncedMemory public_inputs, uint64_t pi_pos, uint64_t n, Intt INTT);

void into_dense_poly_(SyncedMemory public_inputs, SyncedMemory pi_coeffs, SyncedMemory w_r, uint64_t pi_pos, uint64_t n, Intt INTT, cudaStream_t stream1 = (cudaStream_t)0, cudaStream_t stream2 = (cudaStream_t)0);