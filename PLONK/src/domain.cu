#include "PLONK/src/domain.cuh"

Radix2EvaluationDomain::Radix2EvaluationDomain(int num_coeffs):
    size((num_coeffs & (num_coeffs - 1)) == 0 ? num_coeffs : 1 << (32 - __builtin_clz(num_coeffs))),
    group_gen(get_root_of_unity(size)), log_size_of_group(__builtin_ctz(size)), 
    size_as_field_element(fr::make_tensor(size)), size_inv(inv_mod(size_as_field_element)),
    group_gen_inv(inv_mod(group_gen)), generator_inv(inv_mod(fr::GENERATOR()))
{
    SyncedMemory& group_gen_pow = exp_mod(group_gen, size);
    assert(fr::is_equal(group_gen_pow, fr::one()));
}

SyncedMemory& Radix2EvaluationDomain::get_root_of_unity(int n) {
    assert(n > 0 && (n & (n - 1)) == 0);  // n must be a power of 2
    int log_size_of_group = __builtin_ctz(n);  // 获取n的对数值
    assert(log_size_of_group <= fr::TWO_ADICITY);
    SyncedMemory& base = fr::TWO_ADIC_ROOT_OF_UNITY();
    uint64_t exponent = 1ULL << (fr::TWO_ADICITY - log_size_of_group);
    return exp_mod(base, exponent);
}

SyncedMemory& Radix2EvaluationDomain::evaluate_all_lagrange_coefficients(SyncedMemory& tau){
    int size = size;
    SyncedMemory& group_gen = group_gen;
    void* tau_ = tau.mutable_cpu_data();
    SyncedMemory& t_size = exp_mod(tau, size);
    SyncedMemory& z_h_at_tau = sub_mod(t_size, fr::one());

    if (fr::is_equal(z_h_at_tau, fr::zero())) {
        SyncedMemory& u = repeat_zero(size);
        SyncedMemory& omega_i = fr::one();
        void* u_ = u.mutable_cpu_data();
        for (int i = 0; i < size; ++i) {
            if (fr::is_equal(omega_i, tau)) {
                SyncedMemory& one = fr::one();
                const void* one_ = one.cpu_data();
                memcpy((unsigned char*)u_ + i * fr::Limbs * sizeof(uint64_t), one_, one.size());
                break;
            } 
            mul_mod_(omega_i, group_gen);
        }
        return u;
    } else {
        SyncedMemory& f_size = fr::make_tensor(size);
        SyncedMemory& pow_dof = exp_mod(fr::one(), size - 1); 
        SyncedMemory& v_0_inv = mul_mod(f_size, pow_dof);
        SyncedMemory& v_0 = div_mod(fr::one(), v_0_inv);

        void* tau_ = tau.mutable_gpu_data();
        void* v_0_ = v_0.mutable_gpu_data();
        void* group_gen_ = group_gen.mutable_gpu_data();
        void* z_h_at_tau_ = z_h_at_tau.mutable_gpu_data();

        SyncedMemory& coeff_v = gen_sequence(size, group_gen);
        mul_mod_scalar_(coeff_v, v_0);
        SyncedMemory& nominator = mul_mod_scalar(coeff_v, z_h_at_tau);

        SyncedMemory& coeff_r = gen_sequence(size, group_gen);
        SyncedMemory& coeff_tau = repeat_to_poly(tau, size);
        SyncedMemory& denominator = sub_mod(coeff_tau, coeff_r);
        SyncedMemory& denominator_inv = inv_mod(denominator);

        SyncedMemory& lagrange_coefficients = mul_mod(nominator, denominator_inv);

        return lagrange_coefficients;
    }
}

SyncedMemory& Radix2EvaluationDomain::evaluate_vanishing_polynomial(SyncedMemory& tau){
    SyncedMemory& pow_tau = exp_mod(tau, size);
    SyncedMemory& res = sub_mod(pow_tau, fr::one());
    pow_tau.~SyncedMemory();
    return res;
}

SyncedMemory& Radix2EvaluationDomain::element(int i) {
    SyncedMemory group_gen_copy(group_gen.size());
    void* group_gen_copy_ = group_gen_copy.mutable_gpu_data();
    SyncedMemory& coeff = gen_sequence(size, group_gen_copy);
    SyncedMemory res(group_gen.size());
    void* res_ = res.mutable_gpu_data();
    void* coeff_ = coeff.mutable_gpu_data();
    cudaMemcpy(
    res_,
    (uint64_t*)coeff_ + i * fr::Limbs,
    fr::Limbs * sizeof(uint64_t),
    cudaMemcpyDeviceToDevice);
    return res;
}