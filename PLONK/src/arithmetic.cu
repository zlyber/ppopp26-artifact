#include "PLONK/src/arithmetic.cuh"


SyncedMemory& convert_to_bigints(SyncedMemory& p){
    return to_base(p);
}

SyncedMemory& skip_leading_zeros_and_convert_to_bigints(SyncedMemory& p){
    return convert_to_bigints(p);
}

SyncedMemory& poly_add_poly(SyncedMemory& self, SyncedMemory& other) {
    if (self.size() == 0) {
        return other;
    }
    else if (other.size() == 0) {
        return self;
    }
    else if((self.size()>other.size())){
        SyncedMemory result(self.size());
        void* res_gpu = result.mutable_gpu_data();
        void* self_gpu = self.mutable_gpu_data();
        void* other_gpu = other.mutable_gpu_data();
        caffe_gpu_memcpy(self.size(), self_gpu, res_gpu);
        return add_mod(result, other);
    }
    else{
        SyncedMemory result(other.size());
        void* res_gpu = result.mutable_gpu_data();
        void* self_gpu = self.mutable_gpu_data();
        void* other_gpu = other.mutable_gpu_data();
        caffe_gpu_memcpy(other.size(), other_gpu, res_gpu);
        return add_mod(result, self);
    }
}

SyncedMemory& poly_mul_const(SyncedMemory& poly, SyncedMemory& elem) {
    if(poly.size() == 0){
        return poly;
    }
    else{
        return mul_mod_scalar(poly, elem);
    }
}

SyncedMemory& poly_add_poly_mul_const(SyncedMemory& self, SyncedMemory& f, SyncedMemory& other) {
    if (self.size() == 0 && other.size() == 0){
        SyncedMemory res(0);
        return res;
    }

    else if (self.size() == 0){
        void* other_ = other.mutable_gpu_data();
        return mul_mod_scalar(other, f);
    }
    
    else if (other.size() == 0){
        void* self_ = self.mutable_gpu_data();
        return self;
    }
 
    else if (self.size() > other.size()){
        void* self_ = self.mutable_gpu_data();
        void* other_ = other.mutable_gpu_data();
        SyncedMemory& temp = mul_mod_scalar(other, f);
        add_mod_(self, temp);
        return self;
    }

    else {
        void* self_ = self.mutable_gpu_data();
        void* other_ = other.mutable_gpu_data();
        SyncedMemory& mid = pad_poly(self, other.size()/(fr::Limbs*sizeof(uint64_t)));
        SyncedMemory& temp = mul_mod_scalar(other, f);
        SyncedMemory& res = add_mod(temp, self);
        return res;
    }
}
    

void rand_poly(SyncedMemory& poly) {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    uint64_t random = dis(gen);
    SyncedMemory& rand_tensor = fr::make_tensor(random);
    void* tensor = rand_tensor.mutable_cpu_data();
    void* poly_ = poly.mutable_cpu_data();
    memcpy(poly_, tensor, rand_tensor.size());
}


SyncedMemory& compute_first_lagrange_evaluation(int size, SyncedMemory& z_h_eval, SyncedMemory& z_challenge) {
    // single scalar OP on CPU
    SyncedMemory& one = fr::one();
    SyncedMemory& n_fr = fr::make_tensor(size);
    SyncedMemory& z_challenge_sub_one = sub_mod(z_challenge, one);
    SyncedMemory& denom = mul_mod(n_fr , z_challenge_sub_one);
    SyncedMemory& denom_in = div_mod(one , denom);
    return mul_mod(z_h_eval , denom_in);
}

// 多标量乘法
ProjectivePointG1 MSM(SyncedMemory& bases, SyncedMemory& scalar) {
    int min_size = std::min(bases.size(), scalar.size());
    if (min_size == 0) {
        return ProjectivePointG1(fq::one(), fq::one(), fq::zero());
    } 
    else {
        SyncedMemory& commitment = multi_scalar_mult(bases, scalar);
        SyncedMemory x(fq::Limbs * sizeof(uint64_t));
        SyncedMemory y(fq::Limbs * sizeof(uint64_t));
        SyncedMemory z(fq::Limbs * sizeof(uint64_t));
        
        void* x_ = x.mutable_cpu_data();
        void* y_ = y.mutable_cpu_data();
        void* z_ = z.mutable_cpu_data();
        void* commitment_ = commitment.mutable_cpu_data();

        memcpy(x_, commitment_, x.size());
        memcpy(y_, commitment_ + x.size(), x.size());
        memcpy(z_, commitment_ + 2*x.size(), x.size());

        return ProjectivePointG1(x, y, fq::one());
    }
}

