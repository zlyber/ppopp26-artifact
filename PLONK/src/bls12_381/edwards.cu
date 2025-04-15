#include "edwards.cuh"

using namespace caffe;

SyncedMemory COEFF_A(){
    SyncedMemory res(fr::Limbs * sizeof(uint64_t));
    void* res_ = res.mutable_gpu_data();
    uint64_t coeff_a[fr::Limbs] = 
        {
            18446744060824649731UL,
            18102478225614246908UL,
            11073656695919314959UL,
            6613806504683796440UL
        };
    caffe_gpu_memcpy(res.size(), coeff_a, res_);
    return res;
}


SyncedMemory COEFF_D(){
    SyncedMemory res(fr::Limbs * sizeof(uint64_t));
    void* res_ = res.mutable_gpu_data();
    uint64_t coeff_d[fr::Limbs] = 
        {
            3049539848285517488UL,
            18189135023605205683UL,
            8793554888777148625UL,
            6339087681201251886UL,
        };
    caffe_gpu_memcpy(res.size(), coeff_d, res_);
    return res;
}