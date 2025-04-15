#pragma once
#include <cstdint>
#include <cassert>
#include <cstring>
#include "bls12_381/fr.cuh"

class Radix2EvaluationDomain{
    public:
        uint64_t size;
        int log_size_of_group;
        SyncedMemory size_as_field_element;
        SyncedMemory size_inv;
        SyncedMemory group_gen;
        SyncedMemory group_gen_inv;
        SyncedMemory generator_inv;
        Radix2EvaluationDomain(uint64_t n, int log_size, SyncedMemory size_as_field_element,
                               SyncedMemory size_inv, SyncedMemory group_gen, SyncedMemory group_gen_inv,
                               SyncedMemory generator_inv);
};

Radix2EvaluationDomain newdomain(uint64_t num_coeffs);
SyncedMemory get_root_of_unity(int n);
SyncedMemory evaluate_all_lagrange_coefficients(SyncedMemory tau, Radix2EvaluationDomain doamin);
SyncedMemory evaluate_vanishing_polynomial(SyncedMemory tau, Radix2EvaluationDomain domain);
SyncedMemory from_element(uint64_t i, Radix2EvaluationDomain domain);