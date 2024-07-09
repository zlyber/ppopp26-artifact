#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/utils/function.cuh"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/arithmetic.cuh"

class CommitResult{
    public:
        AffinePointG1 commitment;
        SyncedMemory& randomness;
        CommitResult(AffinePointG1 a, SyncedMemory& b);
};

SyncedMemory& empty_randomness();

int calculate_hiding_polynomial_degree(int hiding_bound);

SyncedMemory& randomness_rand(int hiding_bound);

SyncedMemory& rand_add_assign(SyncedMemory& self, SyncedMemory& f, SyncedMemory& other);

CommitResult commit(SyncedMemory& powers_of_g, SyncedMemory& powers_of_gamma_g, SyncedMemory& polynomial, int hiding_bound);