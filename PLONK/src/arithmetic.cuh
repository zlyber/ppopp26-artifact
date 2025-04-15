#pragma once
#include <random>
#include "structure.cuh"

SyncedMemory convert_to_bigints(SyncedMemory p);

SyncedMemory skip_leading_zeros_and_convert_to_bigints(SyncedMemory p);

SyncedMemory poly_add_poly(SyncedMemory self, SyncedMemory other);

SyncedMemory poly_mul_const(SyncedMemory poly, SyncedMemory elem);

SyncedMemory poly_add_poly_mul_const(SyncedMemory self, SyncedMemory f, SyncedMemory other);

void rand_poly(SyncedMemory poly);

SyncedMemory compute_first_lagrange_evaluation(int size, SyncedMemory z_h_eval, SyncedMemory z_challenge);

ProjectivePointG1 MSM(SyncedMemory bases, SyncedMemory scalar, cudaStream_t stream = (cudaStream_t)0);

ProjectivePointG1 to_point(SyncedMemory commitment);

