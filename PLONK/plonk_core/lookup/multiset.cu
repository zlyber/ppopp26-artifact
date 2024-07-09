#include <vector>
#include <unordered_map>
#include "caffe/syncedmem.hpp"
#include <cassert>

struct MultiSet {
    caffe::SyncedMemory elements;
    int Byte_length=8;
    caffe::SyncedMemory& push(caffe::SyncedMemory& element,caffe::SyncedMemory& tail) {
        // elements.push_back(element);
        caffe::SyncedMemory res(elements.size()+element.size());
        void* elements_data = elements.mutable_gpu_data();
        void* res_data = res.mutable_gpu_data();
        void* tail_data = tail.mutable_gpu_data();
        caffe::caffe_gpu_memcpy(elements.size(), elements_data, res_data);
        caffe::caffe_gpu_memcpy(tail.size(),tail_data,res_data+elements.size());
        return res;
    }

    void pad(int n) {
        // 判断 n 是否是 2 的幂
        assert((n & (n - 1)) == 0);

        // 如果 elements 是空的，推入一个 0
        if (elements.size()/Byte_length==0) {
            push(torch::zeros_like(elements[0]));  // 使用第一个元素填充
            elements
        }

        // 如果 size 小于 n，推入 0
        while (elements.size() < n) {
            elements.push_back(elements[0]);  // 使用第一个元素填充
        }
    }

    MultiSet compress(float alpha) {
        auto compress_poly = utils::Multiset_lc(*this, alpha);
        return MultiSet(compress_poly);
    }

    pair<vector<torch::Tensor>, vector<torch::Tensor>> combine_split(MultiSet& f_elements) {
        auto temp_s = from_tensor_list(elements);
        auto temp_f = from_tensor_list(f_elements.elements);
        from_list_gmpy(temp_s);
        from_list_gmpy(temp_f);

        // 创建计数器并初始化
        unordered_map<int, gmpy2::mpz_class> counters;
        for (auto& element : temp_s) {
            counters[element.value] += 1;
        }

        // 将 f 中的元素插入对应的桶中，并检查是否存在对应的 t 中的元素
        for (auto& element : temp_f) {
            if (counters.find(element.value) != counters.end() && counters[element.value] > 0) {
                counters[element.value] += 1;
            } else {
                throw invalid_argument("ElementNotIndexed");
            }
        }

        // 将 s 分成两个交替的半部分：偶数和奇数
        caffe::SyncedMemory  evens;
        caffe::SyncedMemory odds;
        int parity = 0;
        for (auto& pair : counters) {
            auto key = fr::Fr(pair.first);
            auto value = pair.second;
            auto half_count = value / 2;
            for (int i = 0; i < half_count; ++i) {
                evens.push_back(key);
                odds.push_back(key);
            }
            if (value % 2 == 1) {
                if (parity == 1) {
                    odds.push_back(key);
                    parity = 0;
                } else {
                    evens.push_back(key);
                    parity = 1;
                }
            }
        }

        from_gmpy_list(evens);
        from_gmpy_list(odds);
        evens = from_list_tensor(evens);
        odds = from_list_tensor(odds);
        return {evens, odds};
    }
};
