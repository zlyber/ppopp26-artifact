#include "domain.cuh"

Radix2EvaluationDomain::Radix2EvaluationDomain(
    uint64_t n, int log_size, SyncedMemory size_as_field,
    SyncedMemory n_inv, SyncedMemory group_gen_, SyncedMemory group_gen_inv_,
    SyncedMemory generator_inv_):
    size(n), log_size_of_group(log_size),
    size_as_field_element(size_as_field),
    size_inv(n_inv), group_gen(group_gen_), group_gen_inv(group_gen_inv_),
    generator_inv(generator_inv_){}

Radix2EvaluationDomain newdomain(uint64_t num_coeffs)
{
    uint64_t n = (num_coeffs & (num_coeffs - 1)) == 0 ? num_coeffs : 1 << (32 - __builtin_clz(num_coeffs));
    SyncedMemory group_gen_ = get_root_of_unity(n);
    int log_size = __builtin_ctz(n);
    SyncedMemory size_as_field = fr::make_tensor(n);
    SyncedMemory n_inv = inv_mod(size_as_field);
    SyncedMemory group_gen_pow = exp_mod(group_gen_, n);
    SyncedMemory group_gen_inv_ = inv_mod(group_gen_);
    SyncedMemory generator_inv_ = inv_mod(fr::GENERATOR());
    assert(fr::is_equal(group_gen_pow, fr::one()));
    return Radix2EvaluationDomain(
    n, log_size, size_as_field,
    n_inv, group_gen_, group_gen_inv_,
    generator_inv_);
}

SyncedMemory get_root_of_unity(int n) {
    assert(n > 0 && (n & (n - 1)) == 0);  
    int log_size_of_group = __builtin_ctz(n);  
    assert(log_size_of_group <= fr::TWO_ADICITY);
    SyncedMemory base = fr::TWO_ADIC_ROOT_OF_UNITY();
    uint64_t exponent = 1ULL << (fr::TWO_ADICITY - log_size_of_group);
    return exp_mod(base, exponent);
}

SyncedMemory evaluate_all_lagrange_coefficients(SyncedMemory tau, Radix2EvaluationDomain domain){
    uint64_t size = domain.size;
    void* tau_ = tau.mutable_cpu_data();
    SyncedMemory t_size = exp_mod(tau, size);
    SyncedMemory z_h_at_tau = sub_mod(t_size, fr::one());

    if (fr::is_equal(z_h_at_tau, fr::zero())) {
        SyncedMemory u = repeat_zero(size);
        SyncedMemory omega_i = fr::one();
        void* u_ = u.mutable_cpu_data();
        for (int i = 0; i < size; ++i) {
            if (fr::is_equal(omega_i, tau)) {
                SyncedMemory one = fr::one();
                const void* one_ = one.cpu_data();
                memcpy((unsigned char*)u_ + i * fr::Limbs * sizeof(uint64_t), one_, one.size());
                break;
            } 
            mul_mod_(omega_i, domain.group_gen);
        }
        return u;
    } else {
        SyncedMemory f_size = fr::make_tensor(size);
        SyncedMemory pow_dof = exp_mod(fr::one(), size - 1); 
        SyncedMemory v_0_inv = mul_mod(f_size, pow_dof);
        SyncedMemory v_0 = div_mod(fr::one(), v_0_inv);

        void* tau_ = tau.mutable_gpu_data();
        void* v_0_ = v_0.mutable_gpu_data();
        void* group_gen_ = domain.group_gen.mutable_gpu_data();
        void* z_h_at_tau_ = z_h_at_tau.mutable_gpu_data();

        SyncedMemory coeff_v = gen_sequence(size, domain.group_gen);
        mul_mod_scalar_(coeff_v, v_0);
        SyncedMemory nominator = mul_mod_scalar(coeff_v, z_h_at_tau);

        SyncedMemory coeff_r = gen_sequence(size, domain.group_gen);
        SyncedMemory coeff_tau = repeat_to_poly(tau, size);
        SyncedMemory denominator = sub_mod(coeff_tau, coeff_r);
        SyncedMemory denominator_inv = inv_mod(denominator);

        SyncedMemory lagrange_coefficients = mul_mod(nominator, denominator_inv);

        return lagrange_coefficients;
    }
}

SyncedMemory evaluate_vanishing_polynomial(SyncedMemory tau, Radix2EvaluationDomain domain){
    SyncedMemory pow_tau = exp_mod(tau, domain.size);
    SyncedMemory res = sub_mod(pow_tau, fr::one());
    return res;
}

SyncedMemory from_element(uint64_t i, Radix2EvaluationDomain domain) {
    SyncedMemory coeff = gen_sequence(domain.size, domain.group_gen);
    SyncedMemory res(domain.group_gen.size());
    void* res_ = res.mutable_gpu_data();
    void* coeff_ = coeff.mutable_gpu_data();
    caffe_gpu_memcpy(fr::Limbs * sizeof(uint64_t), (uint64_t*)coeff_ + i * fr::Limbs, res_);
    return res;
}