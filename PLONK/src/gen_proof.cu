#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "PLONK/src/domain.cuh"
#include "PLONK/src/transcript/transcript.cuh"
#include "PLONK/src/KZG/kzg10.cuh"

ProofC gen_proof(ProverKeyC pkc, CircuitC csc, CommitKeyC ckc){
    uint64_t size = circuit_bound(csc);
    ProverKey pk = load_pk(pkc, size);
    Circuit cs = load_cs(csc, size);
    CommitKey ck = load_ck(ckc, size);

    Radix2EvaluationDomain domain = Radix2EvaluationDomain(size);
    uint64_t n = domain.size;

    char transcript_init[] = "Merkle tree";
    Transcript transcript = Transcript(transcript_init);
    char pi[] = "pi";
    transcript.append_pi(pi,cs.public_inputs,cs.intended_pi_pos);

    // 1. Compute witness Polynomials
    SyncedMemory& w_l_scalar = pad_poly(cs.w_l, n);
    SyncedMemory& w_r_scalar = pad_poly(cs.w_r, n);
    SyncedMemory& w_o_scalar = pad_poly(cs.w_o, n);
    SyncedMemory& w_4_scalar = pad_poly(cs.w_4, n);

    Intt INTT{fr::TWO_ADICITY};

    SyncedMemory& w_l_poly = INTT.forward(w_l_scalar);
    SyncedMemory& w_r_poly = INTT.forward(w_r_scalar);
    SyncedMemory& w_o_poly = INTT.forward(w_o_scalar);
    SyncedMemory& w_4_poly = INTT.forward(w_4_scalar);

    std::vector<labeldpoly> w_polys;
    w_polys.push_back(labeldpoly(w_l_poly, NULL));
    w_polys.push_back(labeldpoly(w_r_poly, NULL));
    w_polys.push_back(labeldpoly(w_o_poly, NULL));
    w_polys.push_back(labeldpoly(w_4_poly, NULL));

    std::vector<CommitResult> w_commits = commit_poly(ck, w_polys);   
    transcript.append("w_l", w_commits[0].commitment);
    transcript.append("w_r", w_commits[1].commitment);
    transcript.append("w_o", w_commits[2].commitment);
    transcript.append("w_4", w_commits[3].commitment);

    // 2. Derive lookup polynomials

    // Generate table compression factor
    SyncedMemory& zeta = transcript.challenge_scalar("zeta");
    transcript.append("zeta", zeta);
    void* zeta_gpu = zeta.mutable_gpu_data();

    // Compress lookup table into vector of single elements
    SyncedMemory& compressed_t_multiset = compress(pk.lookup_coeffs.table1, pk.lookup_coeffs.table2, pk.lookup_coeffs.table3, pk.lookup_coeffs.table4, zeta);

    // Compute table poly
    SyncedMemory& table_poly = INTT.forward(compressed_t_multiset);

    // Compute query table f
    SyncedMemory& compressed_f_multiset = compute_query_table(
            cs.cs_q_lookup,
            w_l_scalar,
            w_r_scalar,
            w_o_scalar,
            w_4_scalar,
            compressed_t_multiset,
            zeta
        );
    
    // Compute query poly
    SyncedMemory& f_poly = INTT.forward(compressed_f_multiset);
    std::vector<labeldpoly> f_polys;
    f_polys.push_back(labeldpoly(f_poly, NULL));

    // Commit to query polynomial
    std::vector<CommitResult> f_poly_commit = commit_poly(ck, f_polys);

    // Compute s, as the sorted and concatenated version of f and t
    SyncedMemory h_1(compressed_t_multiset.size());
    SyncedMemory h_2(compressed_f_multiset.size());

    void* h_1_gpu = h_1.mutable_gpu_data();
    void* h_2_gpu = h_2.mutable_gpu_data();
    caffe_gpu_memset(h_1.size(), 0, h_1_gpu);
    caffe_gpu_memset(h_2.size(), 0, h_2_gpu);

    // Compute h polys
    SyncedMemory& h_1_poly = INTT.forward(h_1);
    SyncedMemory& h_2_poly = INTT.forward(h_2);

    // Commit to h polys
    std::vector<labeldpoly>h_polys;
    h_polys.push_back(labeldpoly(h_1_poly, NULL));
    h_polys.push_back(labeldpoly(h_2_poly, NULL));
    std::vector<CommitResult>h_commit = commit_poly(ck, h_polys);

    // Add h polynomials to transcript
    transcript.append("h1", h_commit[0].commitment);
    transcript.append("h2", h_commit[1].commitment);

    // 3. Compute permutation polynomial
    // Compute permutation challenge `beta`.
    SyncedMemory& beta = transcript.challenge_scalar("beta");
    transcript.append("beta", beta);
    void* beta_gpu = beta.mutable_gpu_data();

    // Compute permutation challenge `gamma`.
    SyncedMemory& gamma = transcript.challenge_scalar("gamma");
    transcript.append("gamma", gamma);
    void* gamma_gpu = beta.mutable_gpu_data();

    // Compute permutation challenge `delta`.
    SyncedMemory& delta = transcript.challenge_scalar("delta");
    transcript.append("delta", delta);
    void* delta_gpu = beta.mutable_gpu_data();

    // Compute permutation challenge `epsilon`.
    SyncedMemory& epsilon = transcript.challenge_scalar("epsilon");
    transcript.append("epsilon", epsilon);
    void* epsilon_gpu = beta.mutable_gpu_data();

    // Challenges must be different
    assert(!fr::is_equal(beta, gamma) && "challenges must be different");
    assert(!fr::is_equal(beta, delta) && "challenges must be different");
    assert(!fr::is_equal(beta, epsilon) && "challenges must be different");
    assert(!fr::is_equal(gamma, delta) && "challenges must be different");
    assert(!fr::is_equal(gamma, epsilon) && "challenges must be different");
    assert(!fr::is_equal(delta, epsilon) && "challenges must be different");
}