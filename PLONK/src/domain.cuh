#pragma once
#include <cstdint>
#include <cassert>
#include <cstring>
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/utils/function.cuh"

class Radix2EvaluationDomain{
    public:
        int size;
        int log_size_of_group;
        SyncedMemory& size_as_field_element;
        SyncedMemory& size_inv;
        SyncedMemory& group_gen;
        SyncedMemory& group_gen_inv;
        SyncedMemory& generator_inv;

        Radix2EvaluationDomain(int num_coeffs);
        SyncedMemory& get_root_of_unity(int n);
        SyncedMemory& evaluate_all_lagrange_coefficients(SyncedMemory& tau);
        SyncedMemory& evaluate_vanishing_polynomial(SyncedMemory& tau);
        SyncedMemory& element(int i);
};
