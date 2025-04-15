#include <string.h>
#include "PLONK/src/gen_proof.cuh"

extern "C"
ProofC gen_proof(CircuitC csc, ProverKeyC pkc, CommitKeyC ckc){
    return prove(pkc, csc, ckc);
}