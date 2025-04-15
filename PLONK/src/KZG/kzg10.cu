#include "kzg10.cuh"
#include <fstream>

CommitResult::CommitResult(AffinePointG1 a, SyncedMemory b):commitment(a),randomness(b){};

OpenProof::OpenProof(AffinePointG1 w_, SyncedMemory random_v_):w(w_), random_v(random_v_){};

SyncedMemory empty_randomness() {
    SyncedMemory res((size_t)0);
    return res;
}

int calculate_hiding_polynomial_degree(int hiding_bound){
    return hiding_bound + 1;
}

SyncedMemory randomness_rand(int hiding_bound){
    int hiding_poly_degree = calculate_hiding_polynomial_degree(hiding_bound);
    SyncedMemory poly(fr::Limbs * sizeof(uint64_t) * (hiding_poly_degree+1));
    rand_poly(poly);
    return poly;
}

void rand_add_assign(SyncedMemory self, SyncedMemory f, SyncedMemory other){
    void* self_gpu = self.mutable_gpu_data();
    SyncedMemory mid = poly_add_poly_mul_const(self, f, other);
    void* mid_gpu = mid.mutable_gpu_data();
    caffe_gpu_memcpy(mid.size(), mid_gpu, self_gpu);
}

CommitResult commit(SyncedMemory powers_of_g, SyncedMemory powers_of_gamma_g, SyncedMemory polynomial, int hiding_bound){
    SyncedMemory plain_coeffs = skip_leading_zeros_and_convert_to_bigints(polynomial);
    ProjectivePointG1 commitment = MSM(powers_of_g, plain_coeffs);
    SyncedMemory randomness = empty_randomness();
    if (hiding_bound){  
        SyncedMemory randomness = randomness_rand(hiding_bound);
    }
    SyncedMemory random_ints = convert_to_bigints(randomness);
    ProjectivePointG1 random_commitment = MSM(powers_of_gamma_g, random_ints);
    AffinePointG1 random_commitment_affine = to_affine(random_commitment);
    ProjectivePointG1 mix_commitment = add_assign_mixed(commitment, random_commitment_affine);
    AffinePointG1 commitment_affine = to_affine(mix_commitment);
    return CommitResult(commitment_affine, randomness);
}

std::vector<CommitResult> commit_poly(CommitKey ck, std::vector<labeldpoly> polys) {
    srand(42); 
    std::vector<CommitResult> res;

    for (int i = 0; i < polys.size(); ++i) {
        CommitResult partial_res = commit(ck.powers_of_g, ck.powers_of_gamma_g, polys[i].poly, polys[i].hiding_bound);
        res.push_back(partial_res);
    }
    return res;
}

SyncedMemory opening_challenges(SyncedMemory opening_challenge, int exp){
    return exp_mod(opening_challenge, exp);
}

OpenProof open_with_witness_polynomial(
    SyncedMemory powers_of_g,
    SyncedMemory powers_of_gamma_g,
    SyncedMemory point,
    SyncedMemory randomness,
    SyncedMemory witness_polynomial,
    SyncedMemory hiding_witness_polynomial
){
    SyncedMemory witness_coeffs = skip_leading_zeros_and_convert_to_bigints(witness_polynomial);
    ProjectivePointG1 w = MSM(powers_of_g, witness_coeffs);
    SyncedMemory random_v = empty_randomness();
    
    if (hiding_witness_polynomial.size()) {
        SyncedMemory blinding_p = randomness;
        void* point_gpu = point.mutable_gpu_data();
        SyncedMemory blinding_evaluation = evaluate(blinding_p, point);
        SyncedMemory random_witness_coeffs = convert_to_bigints(hiding_witness_polynomial);
        ProjectivePointG1 random_commit = MSM(powers_of_gamma_g, random_witness_coeffs);
        ProjectivePointG1 res = add_assign(w, random_commit);

        return OpenProof(to_affine(res), blinding_evaluation);
    }
    AffinePointG1 res = to_affine(w);
    return OpenProof(res, random_v);
}

witness_poly compute_witness_polynomial(SyncedMemory p, SyncedMemory point, SyncedMemory randomness){
    SyncedMemory mod = fr::MODULUS();
    void* mod_gpu = mod.mutable_gpu_data();
    SyncedMemory neg_p = sub_mod(mod, point);
    SyncedMemory witness_polynomial = poly_div_poly(p, neg_p);
    SyncedMemory random_witness_polynomial((size_t)0);
    if (randomness.size()){
        SyncedMemory random_p = randomness;
        SyncedMemory random_witness_polynomial = poly_div_poly(random_p, neg_p);
        return witness_poly(witness_polynomial, random_witness_polynomial);
    }
    return witness_poly(witness_polynomial, random_witness_polynomial);
}

OpenProof open_proof_internal(SyncedMemory powers_of_g, SyncedMemory powers_of_gamma_g, 
                              SyncedMemory p, SyncedMemory point, SyncedMemory rand){                        
    witness_poly witness = compute_witness_polynomial(p, point, rand);
    OpenProof proof = open_with_witness_polynomial(
        powers_of_g,
        powers_of_gamma_g,
        point,
        rand,
        witness.witness,
        witness.random_witness
    );

    return proof;
}

OpenProof open_proof(CommitKey ck, std::vector<labeldpoly> labeled_polynomials, SyncedMemory point, 
                     SyncedMemory opening_challenge, std::vector<SyncedMemory> rands){
    SyncedMemory combined_polynomial(labeled_polynomials[0].poly.size());
    void* combined_polynomial_gpu = combined_polynomial.mutable_gpu_data();
    SyncedMemory combined_rand(rands[0].size());
    int opening_challenge_counter = 0;
    SyncedMemory curr_challenge = opening_challenges(opening_challenge, opening_challenge_counter);
    void* curr_challenge_gpu = curr_challenge.mutable_gpu_data();
    opening_challenge_counter += 1;

    for(int i = 0; i< labeled_polynomials.size(); i++){
        void* curr_challenge_gpu = curr_challenge.mutable_gpu_data();
        SyncedMemory mid1 = 
        poly_add_poly_mul_const(combined_polynomial, curr_challenge, labeled_polynomials[i].poly);
        void* mid1_gpu = mid1.mutable_gpu_data();
        caffe_gpu_memcpy(combined_polynomial.size(), mid1_gpu, combined_polynomial_gpu);
        mid1 = SyncedMemory();

        rand_add_assign(combined_rand, curr_challenge, rands[i]);
        SyncedMemory mid2 = opening_challenges(opening_challenge, opening_challenge_counter);
        void* mid2_gpu = mid2.mutable_gpu_data();
        caffe_gpu_memcpy(mid2.size(), mid2_gpu, curr_challenge_gpu);
        mid2 = SyncedMemory();
        
        opening_challenge_counter += 1;
    }
    
    OpenProof proof = open_proof_internal(ck.powers_of_g, ck.powers_of_gamma_g, 
                                          combined_polynomial, point, combined_rand);
    return proof;
}