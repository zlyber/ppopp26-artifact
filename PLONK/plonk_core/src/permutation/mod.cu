#include <iostream>
#include "PLONK/utils/function.cuh"
#include "PLONK/utils/function.cu"
#include "PLONK/src/structure.cuh"
#include "caffe/syncedmem.hpp"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/bls12_381/fq.hpp"
#include "PLONK/plonk_core/src/permutation/constants.cu"
#include "PLONK/src/domain.cuh"
#include "PLONK/src/domain.cu"
#define LEN 4 

SyncedMemory& _numerator_irreducible(SyncedMemory& root, SyncedMemory& w, SyncedMemory& k, SyncedMemory& beta, SyncedMemory& gamma) {
    SyncedMemory& mid1 = mul_mod(beta, k); 
    SyncedMemory& mid2 = mul_mod(mid1, root); 
    SyncedMemory& mid3 = add_mod(w, mid2); 
    SyncedMemory& numerator =add_mod(mid3, gamma);
    return numerator; 
}

SyncedMemory& _denominator_irreducible(SyncedMemory& w, SyncedMemory& sigma, SyncedMemory& beta, SyncedMemory& gamma) {
    SyncedMemory& mid1 = mul_mod_scalar(sigma, beta); 
    SyncedMemory& mid2 = add_mod(w, mid1);
    SyncedMemory& denominator = add_mod(mid2, gamma); 
    return denominator; 
}

SyncedMemory& _lookup_ratio(SyncedMemory& one, SyncedMemory& delta, SyncedMemory& epsilon, SyncedMemory& f, SyncedMemory& t, SyncedMemory& t_next,
                  SyncedMemory& h_1, SyncedMemory& h_1_next, SyncedMemory& h_2) {

    SyncedMemory& one_plus_delta =add_mod(delta, one); 
    SyncedMemory& epsilon_one_plus_delta = mul_mod(epsilon, one_plus_delta); 

    SyncedMemory& mid1 = add_mod(epsilon, f); 
    SyncedMemory& mid2 = add_mod(epsilon_one_plus_delta, t); 
    SyncedMemory& mid3 = mul_mod(delta, t_next); 
    SyncedMemory& mid4 = add_mod(mid2, mid3); 
    SyncedMemory& mid5 = mul_mod(one_plus_delta, mid1); 
    SyncedMemory& result = mul_mod(mid4, mid5); 

    SyncedMemory& mid6 = mul_mod(h_2, delta); 
    SyncedMemory& mid7 = add_mod(epsilon_one_plus_delta, h_1); 
    SyncedMemory& mid8 = add_mod(mid6, mid7); 
    SyncedMemory& mid9 = add_mod(epsilon_one_plus_delta, h_2); 
    SyncedMemory& mid10 = mul_mod(h_1_next, delta); 
    SyncedMemory& mid11 = add_mod(mid9, mid10); 
    SyncedMemory& mid12 = mul_mod(mid8, mid11); 
    SyncedMemory&  mid12 = div_mod(one, mid12); 
    SyncedMemory&  result = mul_mod(result, mid12); 

    return result; 
}

SyncedMemory& compute_permutation_poly(const Radix2EvaluationDomain& domain, const std::vector<SyncedMemory&>& wires, SyncedMemory& beta, SyncedMemory& gamma, std::vector<SyncedMemory&> sigma_polys) {
    int n = domain.size;
    SyncedMemory& one = fr::one();
    void* one_gpu_data = one.mutable_gpu_data();
    // Constants defining cosets H, k1H, k2H, etc
    SyncedMemory& obj1 = fr::one();
    SyncedMemory& obj2 = K1();
    SyncedMemory& obj3 = K2();
    SyncedMemory& obj4 = K3();
    void* obj1_gpu_data = obj1.mutable_gpu_data();
    void* obj2_gpu_data = obj2.mutable_gpu_data();
    void* obj3_gpu_data = obj3.mutable_gpu_data();
    void* obj4_gpu_data = obj4.mutable_gpu_data();
    std::vector<SyncedMemory&> ks = {obj1, obj2, obj3, obj4};
    static constexpr int size =fr::TWO_ADICITY;
    Ntt ntt(size);
    SyncedMemory& sigma_mappings0 = ntt.forward(sigma_polys[0]);
    SyncedMemory& sigma_mappings1 = ntt.forward(sigma_polys[1]);
    SyncedMemory& sigma_mappings2 = ntt.forward(sigma_polys[2]);
    SyncedMemory& sigma_mappings3 = ntt.forward(sigma_polys[3]);
    void* sigma_mappings0_gpu_data = sigma_mappings0.mutable_gpu_data();
    void* sigma_mappings1_gpu_data = sigma_mappings1.mutable_gpu_data();
    void* sigma_mappings2_gpu_data = sigma_mappings2.mutable_gpu_data();
    void* sigma_mappings3_gpu_data = sigma_mappings3.mutable_gpu_data();
    std::vector<SyncedMemory&> sigma_mappings = {sigma_mappings0, sigma_mappings1, sigma_mappings2, sigma_mappings3};

    /*
      Transpose wires and sigma values to get "rows" in the form [wl_i,
      wr_i, wo_i, ... ] where each row contains the wire and sigma
      values for a single gate
     Compute all roots, same as calculating twiddles, but doubled in size
    */
    SyncedMemory& domain_group_gen=domain.group_gen;
    void* domain_group_gen_gpu_data = domain_group_gen.mutable_gpu_data();
    SyncedMemory& roots = gen_sequence(n, domain_group_gen);

    SyncedMemory& numerator_product = repeat_to_poly(one, n);
    void* numerator_product_gpu_data = numerator_product.mutable_gpu_data();
    SyncedMemory& denominator_product = repeat_to_poly(one, n);
    void* denominator_product_gpu_data = denominator_product.mutable_gpu_data();

    SyncedMemory& extend_beta = repeat_to_poly(beta, n);
    SyncedMemory& extend_gamma = repeat_to_poly(gamma, n);
    SyncedMemory& extend_one = repeat_to_poly(one, n);

    for (int index = 0; index < ks.size(); ++index) {
        
        SyncedMemory& extend_ks = repeat_to_poly(ks[index], n);
        SyncedMemory& numerator_temps = _numerator_irreducible(roots, wires[index], extend_ks, extend_beta, extend_gamma);
        numerator_product = mul_mod(numerator_temps, numerator_product);
        SyncedMemory& denominator_temps = _denominator_irreducible(wires[index], sigma_mappings[index], beta, extend_gamma);
        denominator_product = mul_mod(denominator_temps, denominator_product);
    }

    SyncedMemory& denominator_product_under = div_mod(extend_one, denominator_product);
    SyncedMemory& gate_coefficient = mul_mod(numerator_product, denominator_product_under);

    SyncedMemory& z = accumulate_mul_poly(gate_coefficient);
    Intt inv_ntt(size);
    SyncedMemory& z_poly=inv_ntt.forward(z);
    return z_poly;
}

SyncedMemory& compute_lookup_permutation_poly(int n,  SyncedMemory& f, SyncedMemory& t,  SyncedMemory& h_1,  SyncedMemory& h_2, SyncedMemory& delta,SyncedMemory& epsilon) {
    int num_f= f.size()/(8*LEN); //8 is the bytes of uint64
    int num_t= t.size()/(8*LEN);
    int num_h_1= h_1.size()/(8*LEN);
    int num_h_2= h_2.size()/(8*LEN);

    assert(num_f== n);
    assert(num_t== n);
    assert(num_h_1 == n);
    assert(num_h_2== n);

    
    SyncedMemory& t_next = repeat_zero(n);
    void* t_next_gpu_data=t_next.mutable_gpu_data();
    void* t_gpu_data=t.mutable_gpu_data();
    caffe_gpu_memcpy(t.size()-1*sizeof(uint64_t),t_gpu_data+sizeof(uint64_t),t_next_gpu_data); //t_next[:n-1]=t[1:]
    caffe_gpu_memcpy(1*sizeof(uint64_t),t_gpu_data,t_next_gpu_data); //t_next[-1]=t[0]

    SyncedMemory& h_1_next = repeat_zero(n);
    void* h_1_next_gpu_data=h_1_next.mutable_gpu_data();
    void* h_1_gpu_data=h_1.mutable_gpu_data();
    caffe_gpu_memcpy(h_1.size()-1*sizeof(uint64_t),h_1_gpu_data+sizeof(uint64_t),h_1_next_gpu_data); //h_1_next[:n-1]=h[1:]
    caffe_gpu_memcpy(1*sizeof(uint64_t),h_1_gpu_data,h_1_next_gpu_data); //h_1_next[-1]=h_1[0]

    SyncedMemory& one = fr::one();
    void* one_gpu_data=one.mutable_gpu_data();

    SyncedMemory& extend_one = repeat_to_poly(one, n);
    SyncedMemory& extend_delta = repeat_to_poly(delta, n);
    SyncedMemory& extend_epsilon =repeat_to_poly(epsilon, n);

    SyncedMemory& product_arguments = _lookup_ratio(extend_one, extend_delta, extend_epsilon, f, t, t_next, h_1, h_1_next, h_2);

    SyncedMemory& p =accumulate_mul_poly(product_arguments);
    Intt inv_ntt(fr::TWO_ADICITY);
    SyncedMemory& p_poly = inv_ntt.forward(p);

    return p_poly;
}