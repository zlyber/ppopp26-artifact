#pragma once
#include "../arithmetic.cuh"
#include <functional>

class CommitResult{
    public:
        AffinePointG1 commitment;
        SyncedMemory randomness;
        CommitResult(AffinePointG1 a, SyncedMemory b);
};

class OpenProof {
    public:
        AffinePointG1 w;
        SyncedMemory random_v;
        OpenProof(AffinePointG1 w_, SyncedMemory random_v_);
};

CommitResult commit(SyncedMemory powers_of_g, SyncedMemory powers_of_gamma_g, SyncedMemory polynomial, int hiding_bound);

std::vector<CommitResult> commit_poly(CommitKey ck, std::vector<labeldpoly> polys);

OpenProof open_proof(CommitKey ck, std::vector<labeldpoly> labeled_polynomials, SyncedMemory point, 
                     SyncedMemory opening_challenge, std::vector<SyncedMemory> rands);

