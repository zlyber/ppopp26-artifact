#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "PLONK/src/domain.cuh"
#include "PLONK/src/transcript/transcript.h"

ProofC gen_proof(ProverKeyC pkc, CircuitC csc, CommitKeyC ckc){
    uint64_t size = circuit_bound(csc);
    ProverKey pk = load_pk(pkc, size);
    Circuit cs = load_cs(csc, size);
    CommitKey ck = load_ck(ckc, size);

    Radix2EvaluationDomain domain = Radix2EvaluationDomain(size);
    uint64_t n = domain.size;

}