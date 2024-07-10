#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/utils/function.cuh"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/serialize.cuh"

ProofC gen_proof(ProverKeyC pkc, CircuitC csc, CommitKeyC ckc){
    uint64_t size = circuit_bound(csc);
    ProverKey pk = load_pk(pkc, size);
    Circuit cs = load_cs(csc, size);
    CommitKey ck = load_ck(ckc, size);

    Radix2EvaluationDomain domain = Radix2EvaluationDomain(size);
    uint64_t n = domain.size;

}