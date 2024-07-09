#include "PLONK/src/KZG/kzg10.cuh"

CommitResult::CommitResult(AffinePointG1 a, SyncedMemory& b):commitment(a),randomness(b){};


SyncedMemory& empty_randomness() {
    SyncedMemory res(0);
    return res;
}

int calculate_hiding_polynomial_degree(int hiding_bound){
    return hiding_bound + 1;
}


SyncedMemory& randomness_rand(int hiding_bound){
    int hiding_poly_degree = calculate_hiding_polynomial_degree(hiding_bound);
    SyncedMemory poly(fr::Limbs * sizeof(uint64_t) * (hiding_poly_degree+1));
    rand_poly(poly);
    return poly;
}

SyncedMemory& rand_add_assign(SyncedMemory& self, SyncedMemory& f, SyncedMemory& other){
    return poly_add_poly_mul_const(self, f, other);
}

CommitResult commit(SyncedMemory& powers_of_g, SyncedMemory& powers_of_gamma_g, SyncedMemory& polynomial, int hiding_bound){
    SyncedMemory& plain_coeffs = skip_leading_zeros_and_convert_to_bigints(polynomial);
    ProjectivePointG1 commitment = MSM(powers_of_g, plain_coeffs);
    SyncedMemory& randomness = empty_randomness();
    if (hiding_bound){
        SyncedMemory& randomness = randomness_rand(hiding_bound);
    }
    SyncedMemory& random_ints = convert_to_bigints(randomness);
    ProjectivePointG1 random_commitment = MSM(powers_of_gamma_g, random_ints);
    AffinePointG1 random_commitment_affine = to_affine(random_commitment);
    ProjectivePointG1 mix_commitment = add_assign_mixed(commitment, random_commitment_affine);
    AffinePointG1 commitment_affine = to_affine(mix_commitment);
    return CommitResult(commitment_affine, randomness);
}