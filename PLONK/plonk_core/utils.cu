
// 线性组合一系列值
// 对于值 [v_0, v_1,... v_k] 返回：
// v_0 + challenge * v_1 + ... + challenge^k * v_k
#include "PLONK/utils/function.cuh"
#include "PLONK/src/structure.cuh"
#include "caffe/syncedmem.hpp"


// 扩展张量函数
// 将输入张量扩展到指定大小，返回扩展后的张量
SyncedMemory& extend_tensor(const caffe::SyncedMemory& input, int size) {
    // 创建一个指定大小和数据类型的全零张量
    Tensor res = torch::zeros({size, 4}, torch::dtype(torch::kBLS12_381_Fr_G1_Mont));
    // 将每个元素设置为输入张量
    for (int i = 0; i < res.size(0); ++i) {
        res[i] = input;
    }
    return res;
}

// 多重集合线性组合函数
// 计算多重集合的线性组合
SyncedMemory& Multiset_lc(caffe::SyncedMemory& values,  caffe::SyncedMemory& challenge) {
    // 获取最后一个值
    void* values_point_gpu= values.mutable_gpu_data();
    caffe::SyncedMemory result(challenge.size());
    uint64_t kth_val = static_cast<uint64_t*>(values_point_gpu)[values.size()-1];
    // 获取反向迭代器（除了最后一个值）
    auto reverse_val = std::vector<caffe::SyncedMemory>(values.rbegin() + 1, values.rend());
    // 线性组合计算
    for (const auto& val : reverse_val) {
        kth_val = mul_mod(kth_val, extended_challenge);
        kth_val = add_mod(kth_val, val);
    }
    return kth_val;
}

// 线性组合函数
// 计算一系列值的线性组合
SyncedMemory& lc(const std::vector<Tensor>& values, const Tensor& challenge) {
    // 获取最后一个值
    Tensor kth_val = values.back();
    // 反向迭代计算
    for (auto it = values.rbegin() + 1; it != values.rend(); ++it) {
        kth_val = mul_mod(kth_val, challenge);
        kth_val = add_mod(kth_val, *it);
    }
    return kth_val;
}
