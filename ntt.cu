#include <iostream>
#include <cstdint>
#include "caffe/interface.hpp"
#include "PLONK/utils/function.cuh"
#include "PLONK/utils/zkp/cuda/zksnark_ntt/ntt_kernel/ntt.cuh"
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <string>
#include <utility>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <vector>
#include <future>
#include <sstream>

static_assert(sizeof(cpu::BLS12_381_Fr_G1) == 4 * sizeof(uint64_t),
              "field element layout assumes four 64-bit limbs");

void read_file(const char* filename, void* data){
   // 打开文件
   std::ifstream file(filename, std::ios::binary);
   if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
   }
   file.seekg(0, std::ios::end);
   size_t fileSize = file.tellg();
   file.seekg(0, std::ios::beg);

   file.read(reinterpret_cast<char*>(data), fileSize);
   if (!file) {
    std::cerr << "Error reading file: " << filename << std::endl;
   }
   file.close();
}

class CpuThreadPool {
public:
    explicit CpuThreadPool(int num_threads) : worker_count_(std::max(1, num_threads)) {
        if (worker_count_ <= 1) {
            return;
        }
        workers_.reserve(worker_count_);
        for (int id = 0; id < worker_count_; ++id) {
            workers_.emplace_back([this, id]() { worker_loop(id); });
        }
    }

    ~CpuThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        task_cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    int worker_count() const {
        return worker_count_;
    }

    template <typename Fn>
    void parallel_for(uint64_t total, Fn fn) {
        if (worker_count_ <= 1 || total < 4096) {
            fn(0, total);
            return;
        }

        std::vector<std::pair<uint64_t, uint64_t>> ranges(worker_count_);
        const uint64_t work_per_worker = total / worker_count_;
        for (int id = 0; id < worker_count_; ++id) {
            const uint64_t begin = id * work_per_worker;
            const uint64_t end = (id == worker_count_ - 1) ? total : begin + work_per_worker;
            ranges[id] = {begin, end};
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_workers_ = worker_count_;
            task_ = [&](int worker_id) {
                const auto [begin, end] = ranges[worker_id];
                if (begin < end) {
                    fn(begin, end);
                }
            };
            ++generation_;
        }
        task_cv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() { return pending_workers_ == 0; });
    }

private:
    void worker_loop(int worker_id) {
        uint64_t seen_generation = 0;
        while (true) {
            std::function<void(int)> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_cv_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
                if (stop_) {
                    return;
                }
                seen_generation = generation_;
                task = task_;
            }

            task(worker_id);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --pending_workers_;
                if (pending_workers_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    int worker_count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    std::function<void(int)> task_;
    uint64_t generation_ = 0;
    int pending_workers_ = 0;
    bool stop_ = false;
};

enum class StageMapPolicy {
    automatic,
    disabled,
    enabled
};

struct StageIndexMap {
    uint64_t chunk_elems = 0;
    uint64_t chunk_count = 0;
    int stage = 0;
    int iterations = 0;
    std::vector<uint32_t> input;
    std::vector<uint32_t> output;

    bool matches(uint64_t elems,
                 uint64_t chunks,
                 int stage_value,
                 int iteration_count) const {
        return chunk_elems == elems &&
               chunk_count == chunks &&
               stage == stage_value &&
               iterations == iteration_count &&
               input.size() == elems * chunks &&
               output.size() == elems * chunks;
    }
};

void ct_stage_indices(uint64_t tid, int stage, int iterations, uint64_t& idx0, uint64_t& idx1) {
    const uint64_t inp_ntt_size = uint64_t{1} << stage;
    const uint64_t out_ntt_size = uint64_t{1} << (stage + iterations - 1);
    const uint64_t thread_ntt_pos = (tid & (out_ntt_size - 1)) >> (iterations - 1);

    idx0 = tid & ~(out_ntt_size - 1);
    idx0 += (tid << stage) & (out_ntt_size - 1);
    idx0 = idx0 * 2 + thread_ntt_pos;
    idx1 = idx0 + inp_ntt_size;
}

void local_stage_indices(uint64_t tid, int iterations, uint64_t& idx0, uint64_t& idx1) {
    const uint64_t out_ntt_size = uint64_t{1} << (iterations - 1);
    const uint64_t thread_ntt_pos = (tid & (out_ntt_size - 1)) >> (iterations - 1);

    idx0 = tid & ~(out_ntt_size - 1);
    idx0 += tid & (out_ntt_size - 1);
    idx0 = idx0 * 2 + thread_ntt_pos;
    idx1 = idx0 + 1;
}

uint64_t rotate_stage_output_index(uint64_t global_idx, int stage, int iterations) {
    const uint64_t mask = ((uint64_t{1} << (stage + iterations)) - (uint64_t{1} << stage));
    uint64_t rotated = global_idx & mask;
    rotated = (rotated >> 1) | (rotated << (iterations - 1));
    return (global_idx & ~mask) | (rotated & mask);
}

StageIndexMap build_stage_index_map(uint64_t chunk_elems,
                                    uint64_t chunk_count,
                                    int stage,
                                    int iterations,
                                    int num_threads) {
    StageIndexMap map;
    map.chunk_elems = chunk_elems;
    map.chunk_count = chunk_count;
    map.stage = stage;
    map.iterations = iterations;
    map.input.resize(chunk_elems * chunk_count);
    map.output.resize(chunk_elems * chunk_count);

    const uint64_t threads_per_chunk = chunk_elems / 2;
    CpuThreadPool pool(num_threads);
    pool.parallel_for(chunk_count * threads_per_chunk, [&](uint64_t begin, uint64_t end) {
        for (uint64_t flat_tid = begin; flat_tid < end; ++flat_tid) {
            const uint64_t chunk = flat_tid / threads_per_chunk;
            const uint64_t local_tid = flat_tid - chunk * threads_per_chunk;
            const uint64_t map_base = chunk * chunk_elems;

            uint64_t global0, global1;
            uint64_t local0, local1;
            ct_stage_indices(flat_tid, stage, iterations, global0, global1);
            local_stage_indices(local_tid, iterations, local0, local1);

            map.input[map_base + local0] = static_cast<uint32_t>(global0);
            map.input[map_base + local1] = static_cast<uint32_t>(global1);
            map.output[map_base + local0] =
                static_cast<uint32_t>(rotate_stage_output_index(global0, stage, iterations));
            map.output[map_base + local1] =
                static_cast<uint32_t>(rotate_stage_output_index(global1, stage, iterations));
        }
    });

    return map;
}

template <typename fr>
void gather_stage_input_map_range(fr* dst,
                                  const fr* src,
                                  const uint32_t* map,
                                  uint64_t begin,
                                  uint64_t end) {
    for (uint64_t i = begin; i < end; ++i) {
        dst[i] = src[map[i]];
    }
}

template <typename fr>
void gather_stage_input_map_parallel(fr* dst,
                                     const fr* src,
                                     const StageIndexMap& map,
                                     uint64_t chunk,
                                     int num_threads,
                                     CpuThreadPool* pool = nullptr) {
    const uint32_t* chunk_map = map.input.data() + chunk * map.chunk_elems;
    if (num_threads <= 1 || map.chunk_elems < 4096) {
        gather_stage_input_map_range(dst, src, chunk_map, 0, map.chunk_elems);
        return;
    }

    if (pool != nullptr && pool->worker_count() == num_threads) {
        pool->parallel_for(map.chunk_elems, [&](uint64_t begin, uint64_t end) {
            gather_stage_input_map_range(dst, src, chunk_map, begin, end);
        });
        return;
    }

    CpuThreadPool local_pool(num_threads);
    local_pool.parallel_for(map.chunk_elems, [&](uint64_t begin, uint64_t end) {
        gather_stage_input_map_range(dst, src, chunk_map, begin, end);
    });
}

template <typename fr>
void scatter_stage_output_map_range(fr* dst,
                                    const fr* src,
                                    const uint32_t* map,
                                    uint64_t begin,
                                    uint64_t end) {
    for (uint64_t i = begin; i < end; ++i) {
        dst[map[i]] = src[i];
    }
}

template <typename fr>
void scatter_stage_output_map_parallel(fr* dst,
                                       const fr* src,
                                       const StageIndexMap& map,
                                       uint64_t chunk,
                                       int num_threads,
                                       CpuThreadPool* pool = nullptr) {
    const uint32_t* chunk_map = map.output.data() + chunk * map.chunk_elems;
    if (num_threads <= 1 || map.chunk_elems < 4096) {
        scatter_stage_output_map_range(dst, src, chunk_map, 0, map.chunk_elems);
        return;
    }

    if (pool != nullptr && pool->worker_count() == num_threads) {
        pool->parallel_for(map.chunk_elems, [&](uint64_t begin, uint64_t end) {
            scatter_stage_output_map_range(dst, src, chunk_map, begin, end);
        });
        return;
    }

    CpuThreadPool local_pool(num_threads);
    local_pool.parallel_for(map.chunk_elems, [&](uint64_t begin, uint64_t end) {
        scatter_stage_output_map_range(dst, src, chunk_map, begin, end);
    });
}

template <typename fr>
void transpose_matrix_range(const fr* input,
                            fr* output,
                            uint64_t rows,
                            uint64_t cols,
                            uint64_t begin,
                            uint64_t end) {
    for (uint64_t idx = begin; idx < end; ++idx) {
        const uint64_t row = idx / cols;
        const uint64_t col = idx - row * cols;
        output[col * rows + row] = input[idx];
    }
}

template <typename fr>
void transpose_matrix_parallel(const fr* input,
                               fr* output,
                               uint64_t rows,
                               uint64_t cols,
                               int num_threads) {
    const uint64_t total = rows * cols;
    CpuThreadPool pool(num_threads);
    pool.parallel_for(total, [&](uint64_t begin, uint64_t end) {
        transpose_matrix_range(input, output, rows, cols, begin, end);
    });
}

template <typename fr>
void permute_stage_input_chunk_range(fr* dst,
                                     const fr* src,
                                     uint64_t chunk,
                                     uint64_t chunk_elems,
                                     int stage,
                                     int iterations,
                                     uint64_t begin,
                                     uint64_t end) {
    const uint64_t tid_base = chunk * (chunk_elems / 2);
    for (uint64_t local_tid = begin; local_tid < end; ++local_tid) {
        uint64_t global0, global1;
        uint64_t local0, local1;
        ct_stage_indices(tid_base + local_tid, stage, iterations, global0, global1);
        local_stage_indices(local_tid, iterations, local0, local1);

        dst[local0] = src[global0];
        dst[local1] = src[global1];
    }
}

template <typename fr>
void permute_stage_input_chunk_parallel(fr* dst,
                                        const fr* src,
                                        uint64_t chunk,
                                        uint64_t chunk_elems,
                                        int stage,
                                        int iterations,
                                        int num_threads,
                                        CpuThreadPool* pool = nullptr) {
    const uint64_t threads_per_chunk = chunk_elems / 2;
    if (num_threads <= 1 || threads_per_chunk < 4096) {
        permute_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                        stage, iterations, 0, threads_per_chunk);
        return;
    }

    if (pool != nullptr && pool->worker_count() == num_threads) {
        pool->parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
            permute_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                            stage, iterations, begin, end);
        });
        return;
    }

    CpuThreadPool local_pool(num_threads);
    local_pool.parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
        permute_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                        stage, iterations, begin, end);
    });
}

__device__ uint64_t reverse_bits_device(uint64_t value, int bits) {
    uint64_t reversed = 0;
    for (int bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

__global__ void block_bit_reverse_fields_kernel(uint64_t* data,
                                                uint64_t block_elems,
                                                uint64_t batch,
                                                int lg_block_elems) {
    const uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t total = block_elems * batch;
    if (idx >= total) {
        return;
    }

    const uint64_t block = idx / block_elems;
    const uint64_t local = idx - block * block_elems;
    const uint64_t rev = reverse_bits_device(local, lg_block_elems);
    if (local >= rev) {
        return;
    }

    const uint64_t a = (block * block_elems + local) * 4;
    const uint64_t b = (block * block_elems + rev) * 4;
    uint64_t tmp0 = data[a];
    uint64_t tmp1 = data[a + 1];
    uint64_t tmp2 = data[a + 2];
    uint64_t tmp3 = data[a + 3];

    data[a] = data[b];
    data[a + 1] = data[b + 1];
    data[a + 2] = data[b + 2];
    data[a + 3] = data[b + 3];

    data[b] = tmp0;
    data[b + 1] = tmp1;
    data[b + 2] = tmp2;
    data[b + 3] = tmp3;
}

void launch_block_bit_reverse(void* data,
                              uint64_t block_elems,
                              uint64_t batch,
                              int lg_block_elems,
                              cudaStream_t stream) {
    const uint64_t total = block_elems * batch;
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    block_bit_reverse_fields_kernel<<<static_cast<unsigned int>(grid_size),
                                      block_size, 0, stream>>>(
        reinterpret_cast<uint64_t*>(data),
        block_elems,
        batch,
        lg_block_elems);
}

__global__ void transpose_fields_kernel(uint64_t* dst,
                                        const uint64_t* src,
                                        uint64_t rows,
                                        uint64_t cols) {
    const uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t total = rows * cols;
    if (idx >= total) {
        return;
    }

    const uint64_t row = idx / cols;
    const uint64_t col = idx - row * cols;
    const uint64_t out = col * rows + row;
    const uint64_t src_word = idx * 4;
    const uint64_t dst_word = out * 4;

    dst[dst_word] = src[src_word];
    dst[dst_word + 1] = src[src_word + 1];
    dst[dst_word + 2] = src[src_word + 2];
    dst[dst_word + 3] = src[src_word + 3];
}

void launch_transpose_fields(void* dst,
                             const void* src,
                             uint64_t rows,
                             uint64_t cols,
                             cudaStream_t stream) {
    const uint64_t total = rows * cols;
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    transpose_fields_kernel<<<static_cast<unsigned int>(grid_size),
                              block_size, 0, stream>>>(
        reinterpret_cast<uint64_t*>(dst),
        reinterpret_cast<const uint64_t*>(src),
        rows,
        cols);
}

__global__ void standard4_twiddle_kernel(cuda::fr* data,
                                         uint64_t rows,
                                         uint64_t cols,
                                         uint64_t col_base,
                                         int lg_N,
                                         const cuda::fr* partial_twiddles) {
    const uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t total = rows * cols;
    if (idx >= total) {
        return;
    }

    const uint64_t local_col = idx / rows;
    const uint64_t row = idx - local_col * rows;
    const uint64_t col = col_base + local_col;
    const uint64_t shift = MAX_LG_DOMAIN_SIZE - lg_N;
    const uint64_t exponent = (row * col) << shift;
    data[idx] *= cuda::get_intermediate_root(static_cast<cuda::index_t>(exponent),
                                             partial_twiddles);
}

void launch_standard4_twiddle(void* data,
                              uint64_t rows,
                              uint64_t cols,
                              uint64_t col_base,
                              int lg_N,
                              Ntt& ntt,
                              cudaStream_t stream) {
    auto* params = reinterpret_cast<cuda::fr*>(ntt.Params.mutable_gpu_data());
    cuda::fr* partial_twiddles = params;
    const uint64_t total = rows * cols;
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    standard4_twiddle_kernel<<<static_cast<unsigned int>(grid_size),
                               block_size, 0, stream>>>(
        reinterpret_cast<cuda::fr*>(data),
        rows,
        cols,
        col_base,
        lg_N,
                               partial_twiddles);
}

__global__ void batched_ct_stage_kernel(cuda::fr* data,
                                        uint64_t len,
                                        uint64_t batch,
                                        int lg_len,
                                        int stage,
                                        const cuda::fr* partial_twiddles) {
    const uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t pairs_per_batch = len / 2;
    const uint64_t total = pairs_per_batch * batch;
    if (idx >= total) {
        return;
    }

    const uint64_t item = idx / pairs_per_batch;
    const uint64_t pair = idx - item * pairs_per_batch;
    const uint64_t half = uint64_t{1} << stage;
    const uint64_t m = half << 1;
    const uint64_t group = pair / half;
    const uint64_t j = pair - group * half;
    const uint64_t base = item * len + group * m + j;
    const uint64_t shift = MAX_LG_DOMAIN_SIZE - lg_len;
    const uint64_t exponent = (j << (lg_len - stage - 1)) << shift;

    cuda::fr root = cuda::get_intermediate_root(static_cast<cuda::index_t>(exponent),
                                                partial_twiddles);
    cuda::fr u = data[base];
    cuda::fr v = data[base + half] * root;
    data[base] = u + v;
    data[base + half] = u - v;
}

void launch_batched_ct_stage(void* data,
                             uint64_t batch,
                             uint64_t len,
                             int lg_len,
                             int stage,
                             Ntt& ntt,
                             cudaStream_t stream) {
    auto* params = reinterpret_cast<cuda::fr*>(ntt.Params.mutable_gpu_data());
    cuda::fr* partial_twiddles = params;
    const uint64_t total = batch * (len / 2);
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    batched_ct_stage_kernel<<<static_cast<unsigned int>(grid_size),
                              block_size, 0, stream>>>(
        reinterpret_cast<cuda::fr*>(data),
        len,
        batch,
        lg_len,
        stage,
        partial_twiddles);
}

void run_batched_simple_ntt(void* data,
                            uint64_t batch,
                            uint64_t len,
                            int lg_len,
                            Ntt& ntt,
                            cudaStream_t stream) {
    launch_block_bit_reverse(data, len, batch, lg_len, stream);
    for (int stage = 0; stage < lg_len; ++stage) {
        launch_batched_ct_stage(data, batch, len, lg_len, stage, ntt, stream);
    }
}

template <typename fr>
void scatter_stage_output_chunk_range(fr* dst,
                                      const fr* src,
                                      uint64_t chunk,
                                      uint64_t chunk_elems,
                                      int stage,
                                      int iterations,
                                      uint64_t begin,
                                      uint64_t end) {
    for (uint64_t local_tid = begin; local_tid < end; ++local_tid) {
        uint64_t global0, global1;
        uint64_t local0, local1;
        ct_stage_indices(chunk * (chunk_elems / 2) + local_tid, stage, iterations, global0, global1);
        local_stage_indices(local_tid, iterations, local0, local1);

        const uint64_t out0 = rotate_stage_output_index(global0, stage, iterations);
        const uint64_t out1 = rotate_stage_output_index(global1, stage, iterations);

        dst[out0] = src[local0];
        dst[out1] = src[local1];
    }
}

template <typename fr>
void scatter_stage_output_chunk_parallel(fr* dst,
                                         const fr* src,
                                         uint64_t chunk,
                                         uint64_t chunk_elems,
                                         int stage,
                                         int iterations,
                                         int num_threads,
                                         CpuThreadPool* pool = nullptr) {
    const uint64_t threads_per_chunk = chunk_elems / 2;
    if (num_threads <= 1 || threads_per_chunk < 4096) {
        scatter_stage_output_chunk_range(dst, src, chunk, chunk_elems,
                                         stage, iterations, 0, threads_per_chunk);
        return;
    }

    if (pool != nullptr && pool->worker_count() == num_threads) {
        pool->parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
            scatter_stage_output_chunk_range(dst, src, chunk, chunk_elems,
                                             stage, iterations, begin, end);
        });
        return;
    }

    CpuThreadPool local_pool(num_threads);
    local_pool.parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
        scatter_stage_output_chunk_range(dst, src, chunk, chunk_elems,
                                         stage, iterations, begin, end);
    });
}

__global__ void scatter_stage_kernel(uint64_t* dst,
                                     const uint64_t* src,
                                     uint64_t chunk,
                                     uint64_t chunk_elems,
                                     int stage,
                                     int iterations) {
    uint64_t local_tid = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t threads_per_chunk = chunk_elems / 2;
    if (local_tid >= threads_per_chunk) {
        return;
    }

    const uint64_t global_tid = chunk * threads_per_chunk + local_tid;
    const uint64_t out_ntt_size = uint64_t{1} << (stage + iterations - 1);
    const uint64_t inp_ntt_size = uint64_t{1} << stage;
    const uint64_t thread_ntt_pos = (global_tid & (out_ntt_size - 1)) >> (iterations - 1);

    uint64_t global0 = global_tid & ~(out_ntt_size - 1);
    global0 += (global_tid << stage) & (out_ntt_size - 1);
    global0 = global0 * 2 + thread_ntt_pos;
    uint64_t global1 = global0 + inp_ntt_size;

    const uint64_t local_out_ntt_size = uint64_t{1} << (iterations - 1);
    const uint64_t local_thread_ntt_pos = (local_tid & (local_out_ntt_size - 1)) >> (iterations - 1);

    uint64_t local0 = local_tid & ~(local_out_ntt_size - 1);
    local0 += local_tid & (local_out_ntt_size - 1);
    local0 = local0 * 2 + local_thread_ntt_pos;
    uint64_t local1 = local0 + 1;

    uint64_t mask = ((uint64_t{1} << (stage + iterations)) - (uint64_t{1} << stage));
    uint64_t rotw0 = global0 & mask;
    rotw0 = (rotw0 >> 1) | (rotw0 << (iterations - 1));
    uint64_t out0 = ((global0 & ~mask) | (rotw0 & mask)) - chunk * chunk_elems;

    uint64_t rotw1 = global1 & mask;
    rotw1 = (rotw1 >> 1) | (rotw1 << (iterations - 1));
    uint64_t out1 = ((global1 & ~mask) | (rotw1 & mask)) - chunk * chunk_elems;

    const uint64_t src0 = local0 * 4;
    const uint64_t src1 = local1 * 4;
    const uint64_t dst0 = out0 * 4;
    const uint64_t dst1 = out1 * 4;

    dst[dst0] = src[src0];
    dst[dst0 + 1] = src[src0 + 1];
    dst[dst0 + 2] = src[src0 + 2];
    dst[dst0 + 3] = src[src0 + 3];

    dst[dst1] = src[src1];
    dst[dst1 + 1] = src[src1 + 1];
    dst[dst1 + 2] = src[src1 + 2];
    dst[dst1 + 3] = src[src1 + 3];
}

__global__ void transition_stage_kernel(uint64_t* dst,
                                        const uint64_t* src,
                                        uint64_t chunk,
                                        uint64_t chunk_elems,
                                        int prev_stage,
                                        int prev_iterations,
                                        int next_stage,
                                        int next_iterations) {
    uint64_t local_tid = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t threads_per_chunk = chunk_elems / 2;
    if (local_tid >= threads_per_chunk) {
        return;
    }

    const uint64_t global_tid = chunk * threads_per_chunk + local_tid;
    const uint64_t out_ntt_size = uint64_t{1} << (next_stage + next_iterations - 1);
    const uint64_t inp_ntt_size = uint64_t{1} << next_stage;
    const uint64_t thread_ntt_pos = (global_tid & (out_ntt_size - 1)) >> (next_iterations - 1);

    uint64_t global0 = global_tid & ~(out_ntt_size - 1);
    global0 += (global_tid << next_stage) & (out_ntt_size - 1);
    global0 = global0 * 2 + thread_ntt_pos;
    uint64_t global1 = global0 + inp_ntt_size;

    const uint64_t local_out_ntt_size = uint64_t{1} << (next_iterations - 1);
    const uint64_t local_thread_ntt_pos = (local_tid & (local_out_ntt_size - 1)) >> (next_iterations - 1);

    uint64_t local0 = local_tid & ~(local_out_ntt_size - 1);
    local0 += local_tid & (local_out_ntt_size - 1);
    local0 = local0 * 2 + local_thread_ntt_pos;
    uint64_t local1 = local0 + 1;

    uint64_t mask = ((uint64_t{1} << (prev_stage + prev_iterations)) - (uint64_t{1} << prev_stage));
    uint64_t rotw0 = global0 & mask;
    rotw0 = ((rotw0 << 1) | (rotw0 >> (prev_iterations - 1))) & mask;
    uint64_t src0 = ((global0 & ~mask) | rotw0) - chunk * chunk_elems;

    uint64_t rotw1 = global1 & mask;
    rotw1 = ((rotw1 << 1) | (rotw1 >> (prev_iterations - 1))) & mask;
    uint64_t src1 = ((global1 & ~mask) | rotw1) - chunk * chunk_elems;

    const uint64_t src_word0 = src0 * 4;
    const uint64_t src_word1 = src1 * 4;
    const uint64_t dst_word0 = local0 * 4;
    const uint64_t dst_word1 = local1 * 4;

    dst[dst_word0] = src[src_word0];
    dst[dst_word0 + 1] = src[src_word0 + 1];
    dst[dst_word0 + 2] = src[src_word0 + 2];
    dst[dst_word0 + 3] = src[src_word0 + 3];

    dst[dst_word1] = src[src_word1];
    dst[dst_word1 + 1] = src[src_word1 + 1];
    dst[dst_word1 + 2] = src[src_word1 + 2];
    dst[dst_word1 + 3] = src[src_word1 + 3];
}

void launch_scatter_stage(void* dst,
                          const void* src,
                          uint64_t chunk,
                          uint64_t chunk_elems,
                          int stage,
                          int iterations,
                          cudaStream_t stream) {
    const int block = 256;
    const uint64_t threads = chunk_elems / 2;
    const int grid = static_cast<int>((threads + block - 1) / block);
    scatter_stage_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint64_t*>(dst),
        reinterpret_cast<const uint64_t*>(src),
        chunk,
        chunk_elems,
        stage,
        iterations);
}

void launch_transition_stage(void* dst,
                             const void* src,
                             uint64_t chunk,
                             uint64_t chunk_elems,
                             int prev_stage,
                             int prev_iterations,
                             int next_stage,
                             int next_iterations,
                             cudaStream_t stream) {
    const int block = 256;
    const uint64_t threads = chunk_elems / 2;
    const int grid = static_cast<int>((threads + block - 1) / block);
    transition_stage_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint64_t*>(dst),
        reinterpret_cast<const uint64_t*>(src),
        chunk,
        chunk_elems,
        prev_stage,
        prev_iterations,
        next_stage,
        next_iterations);
}

void run_first_ntt_pass(cpu::BLS12_381_Fr_G1* stage_in,
                        cpu::BLS12_381_Fr_G1* stage_out,
                        uint64_t chunk_elems,
                        uint64_t chunk_count,
                        int first_iterations,
                        int lg_N,
                        Ntt ntt) {
    const int pipeline_slots = 2;
    const size_t chunk_bytes = chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);

    SyncedMemory d_buffers[pipeline_slots] = {
        SyncedMemory(chunk_bytes),
        SyncedMemory(chunk_bytes)
    };
    SyncedMemory d_scratch[pipeline_slots] = {
        SyncedMemory(chunk_bytes),
        SyncedMemory(chunk_bytes)
    };
    void* d_ptrs[pipeline_slots];
    void* scratch_ptrs[pipeline_slots];
    cudaStream_t streams[pipeline_slots];
    cudaEvent_t done[pipeline_slots];

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        d_ptrs[slot] = d_buffers[slot].mutable_gpu_data();
        scratch_ptrs[slot] = d_scratch[slot].mutable_gpu_data();
        cudaStreamCreate(&streams[slot]);
        cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming);
    }

    auto* in_bytes = reinterpret_cast<char*>(stage_in);
    auto* out_bytes = reinterpret_cast<char*>(stage_out);
    for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int slot = chunk % pipeline_slots;
        if (chunk >= pipeline_slots) {
            cudaEventSynchronize(done[slot]);
        }

        cudaMemcpyAsync(d_ptrs[slot], in_bytes + chunk * chunk_bytes,
                        chunk_bytes, cudaMemcpyHostToDevice, streams[slot]);
        const int first_step = first_iterations / 2;
        const int second_step = first_iterations - first_step;
        ntt.stage_no_rotate(d_buffers[slot], lg_N, first_step, 0, chunk, streams[slot]);
        launch_transition_stage(scratch_ptrs[slot], d_ptrs[slot], chunk, chunk_elems,
                                0, first_step, first_step, second_step, streams[slot]);
        ntt.stage_no_rotate(d_scratch[slot], lg_N, second_step, first_step, chunk, streams[slot]);
        launch_scatter_stage(d_ptrs[slot], scratch_ptrs[slot], chunk, chunk_elems,
                             first_step, second_step, streams[slot]);
        cudaMemcpyAsync(out_bytes + chunk * chunk_bytes, d_ptrs[slot],
                        chunk_bytes, cudaMemcpyDeviceToHost, streams[slot]);
        cudaEventRecord(done[slot], streams[slot]);
    }

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        cudaEventSynchronize(done[slot]);
        cudaEventDestroy(done[slot]);
        cudaStreamDestroy(streams[slot]);
    }
}

void run_second_ntt_pass_async_scatter(cpu::BLS12_381_Fr_G1* stage_in,
                                       cpu::BLS12_381_Fr_G1* output,
                                       uint64_t chunk_elems,
                                       uint64_t chunk_count,
                                       int stage,
                                       int iterations,
                                       int lg_N,
                                       int num_threads,
                                       Ntt ntt,
                                       const StageIndexMap* stage_map = nullptr) {
    const int pipeline_slots = 2;
    const size_t chunk_bytes = chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);

    SyncedMemory h_in[pipeline_slots] = {
        SyncedMemory(chunk_bytes),
        SyncedMemory(chunk_bytes)
    };
    SyncedMemory h_out[pipeline_slots] = {
        SyncedMemory(chunk_bytes),
        SyncedMemory(chunk_bytes)
    };
    SyncedMemory d_buffers[pipeline_slots] = {
        SyncedMemory(chunk_bytes),
        SyncedMemory(chunk_bytes)
    };

    cpu::BLS12_381_Fr_G1* h_in_ptrs[pipeline_slots];
    cpu::BLS12_381_Fr_G1* h_out_ptrs[pipeline_slots];
    void* d_ptrs[pipeline_slots];
    cudaStream_t streams[pipeline_slots];
    cudaEvent_t done[pipeline_slots];
    uint64_t completed_chunk[pipeline_slots] = {};
    bool slot_valid[pipeline_slots] = {};
    bool scatter_pending[pipeline_slots] = {};
    std::future<void> scatter_tasks[pipeline_slots];
    CpuThreadPool gather_pool(num_threads);
    CpuThreadPool scatter_pool0(num_threads);
    CpuThreadPool scatter_pool1(num_threads);
    CpuThreadPool* scatter_pools[pipeline_slots] = {&scatter_pool0, &scatter_pool1};

    const bool use_map = stage_map != nullptr &&
                         stage_map->matches(chunk_elems, chunk_count, stage, iterations);

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        h_in_ptrs[slot] = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(h_in[slot].mutable_cpu_data());
        h_out_ptrs[slot] = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(h_out[slot].mutable_cpu_data());
        d_ptrs[slot] = d_buffers[slot].mutable_gpu_data();
        cudaStreamCreate(&streams[slot]);
        cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming);
    }

    auto wait_scatter = [&](int slot) {
        if (scatter_pending[slot]) {
            scatter_tasks[slot].get();
            scatter_pending[slot] = false;
        }
    };

    auto launch_scatter = [&](int slot, uint64_t chunk) {
        wait_scatter(slot);
        scatter_tasks[slot] = std::async(std::launch::async, [&, slot, chunk]() {
            if (use_map) {
                scatter_stage_output_map_parallel(output, h_out_ptrs[slot],
                                                  *stage_map, chunk,
                                                  num_threads, scatter_pools[slot]);
            } else {
                scatter_stage_output_chunk_parallel(output, h_out_ptrs[slot],
                                                    chunk, chunk_elems,
                                                    stage, iterations,
                                                    num_threads, scatter_pools[slot]);
            }
        });
        scatter_pending[slot] = true;
    };

    for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int slot = chunk % pipeline_slots;
        if (slot_valid[slot]) {
            cudaEventSynchronize(done[slot]);
            launch_scatter(slot, completed_chunk[slot]);
            slot_valid[slot] = false;
        }

        if (use_map) {
            gather_stage_input_map_parallel(h_in_ptrs[slot], stage_in, *stage_map,
                                            chunk, num_threads, &gather_pool);
        } else {
            permute_stage_input_chunk_parallel(h_in_ptrs[slot], stage_in, chunk, chunk_elems,
                                               stage, iterations, num_threads, &gather_pool);
        }
        cudaMemcpyAsync(d_ptrs[slot], h_in_ptrs[slot], chunk_bytes,
                        cudaMemcpyHostToDevice, streams[slot]);
        ntt.stage_no_rotate(d_buffers[slot], lg_N, iterations, stage, chunk, streams[slot]);
        wait_scatter(slot);
        cudaMemcpyAsync(h_out_ptrs[slot], d_ptrs[slot], chunk_bytes,
                        cudaMemcpyDeviceToHost, streams[slot]);
        cudaEventRecord(done[slot], streams[slot]);
        completed_chunk[slot] = chunk;
        slot_valid[slot] = true;
    }

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        if (slot_valid[slot]) {
            cudaEventSynchronize(done[slot]);
            launch_scatter(slot, completed_chunk[slot]);
            slot_valid[slot] = false;
        }
    }
    for (int slot = 0; slot < pipeline_slots; ++slot) {
        wait_scatter(slot);
    }

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        cudaEventDestroy(done[slot]);
        cudaStreamDestroy(streams[slot]);
    }
}

void our_ntt(SyncedMemory input,
             int lg_N,
             int lg_i,
             int lg_j,
             int lg_k,
             int step1_thread,
             int step2_thread,
             Ntt ntt,
             const StageIndexMap* second_stage_map = nullptr)
{
    (void)lg_j;
    uint64_t N = uint64_t{1} << lg_N;
    uint64_t I = uint64_t{1} << lg_i;
    uint64_t k = uint64_t{1} << lg_k;
    uint64_t chunk_elems = I * k;
    uint64_t chunk_count = N / chunk_elems;

    SyncedMemory stage1_in(input.size());
    SyncedMemory stage1_out(input.size());

    auto* input_cpu = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input.mutable_cpu_data());
    auto* stage1_in_cpu = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(stage1_in.mutable_cpu_data());
    auto* stage1_out_cpu = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(stage1_out.mutable_cpu_data());

    const int second_iterations = lg_N - lg_i;
    if (second_iterations > 10) {
        std::cerr << "unsupported NTT split: second stage is "
                  << second_iterations << " bits; expected <= 10" << std::endl;
        std::exit(1);
    }
    uint32_t* step1_map = new uint32_t[N];
    calculate_map1(step1_map, lg_N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    bit_rev_step1_parallel(stage1_in_cpu, input_cpu, step1_map, lg_N, step1_thread, 0);
    run_first_ntt_pass(stage1_in_cpu, stage1_out_cpu,
                       chunk_elems, chunk_count, lg_i, lg_N, ntt);
    run_second_ntt_pass_async_scatter(stage1_out_cpu, input_cpu, chunk_elems, chunk_count,
                                      lg_i, second_iterations, lg_N, step2_thread, ntt,
                                      second_stage_map);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "our-ntt execution time: " << milliseconds<< " ms" << std::endl;
    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] step1_map;
}

void standard4_ntt(SyncedMemory input,
                   int lg_N,
                   int lg_i,
                   int lg_j,
                   int lg_k,
                   int transpose_threads,
                   Ntt ntt) {
    const uint64_t N = uint64_t{1} << lg_N;
    const uint64_t I = uint64_t{1} << lg_i;
    const uint64_t J = uint64_t{1} << lg_j;
    const uint64_t k = uint64_t{1} << lg_k;
    const size_t field_bytes = sizeof(cpu::BLS12_381_Fr_G1);
    const uint64_t first_chunk_elems = I * k;
    const uint64_t second_chunk_elems = J * k;

    SyncedMemory d_col_tile(first_chunk_elems * field_bytes);
    SyncedMemory d_col_batch(first_chunk_elems * field_bytes);
    SyncedMemory d_row_tile(second_chunk_elems * field_bytes);
    SyncedMemory transposed(input.size());

    auto* input_cpu = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(input.mutable_cpu_data());
    auto* transpose_cpu = reinterpret_cast<cpu::BLS12_381_Fr_G1*>(transposed.mutable_cpu_data());
    void* d_col_tile_ptr = d_col_tile.mutable_gpu_data();
    void* d_col_batch_ptr = d_col_batch.mutable_gpu_data();
    void* d_row_tile_ptr = d_row_tile.mutable_gpu_data();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        cudaMemcpy2DAsync(d_col_tile_ptr,
                          k * field_bytes,
                          input_cpu,
                          J * field_bytes,
                          k * field_bytes,
                          I,
                          cudaMemcpyHostToDevice,
                          stream);
        launch_transpose_fields(d_col_batch_ptr, d_col_tile_ptr, I, k, stream);
        run_batched_simple_ntt(d_col_batch_ptr, k, I, lg_i, ntt, stream);
        launch_standard4_twiddle(d_col_batch_ptr, I, k, 0, lg_N, ntt, stream);
        cudaStreamSynchronize(stream);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    for (uint64_t col_base = 0; col_base < J; col_base += k) {
        cudaMemcpy2DAsync(d_col_tile_ptr,
                          k * field_bytes,
                          input_cpu + col_base,
                          J * field_bytes,
                          k * field_bytes,
                          I,
                          cudaMemcpyHostToDevice,
                          stream);
        launch_transpose_fields(d_col_batch_ptr, d_col_tile_ptr, I, k, stream);
        run_batched_simple_ntt(d_col_batch_ptr, k, I, lg_i, ntt, stream);
        launch_standard4_twiddle(d_col_batch_ptr, I, k, col_base, lg_N, ntt, stream);
        launch_transpose_fields(d_col_tile_ptr, d_col_batch_ptr, k, I, stream);
        cudaMemcpy2DAsync(input_cpu + col_base,
                          J * field_bytes,
                          d_col_tile_ptr,
                          k * field_bytes,
                          k * field_bytes,
                          I,
                          cudaMemcpyDeviceToHost,
                          stream);
        cudaStreamSynchronize(stream);
    }

    for (uint64_t row_base = 0; row_base < I; row_base += k) {
        auto* row_ptr = input_cpu + row_base * J;
        const size_t row_chunk_bytes = second_chunk_elems * field_bytes;
        cudaMemcpyAsync(d_row_tile_ptr, row_ptr, row_chunk_bytes,
                        cudaMemcpyHostToDevice, stream);
        run_batched_simple_ntt(d_row_tile_ptr, k, J, lg_j, ntt, stream);
        cudaMemcpyAsync(row_ptr, d_row_tile_ptr, row_chunk_bytes,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    transpose_matrix_parallel(input_cpu, transpose_cpu, I, J, transpose_threads);
    std::memcpy(input_cpu, transpose_cpu, N * field_bytes);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "standard4-ntt execution time: " << milliseconds << " ms" << std::endl;
    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int step2_threads_for_lg_k(int lg_k, int lg_N = 24) {
    const char* override = std::getenv("STEP2_THREADS");
    if (override != nullptr) {
        return std::max(1, std::atoi(override));
    }
    if (lg_k >= 1 && lg_k <= 8) {
        static const int tuned_22[8] = {4, 8, 8, 8, 12, 12, 12, 12};
        static const int tuned_24[8] = {8, 8, 12, 12, 20, 12, 12, 20};
        static const int tuned_26[8] = {16, 12, 12, 12, 20, 20, 20, 20};
        if (lg_N == 22) {
            return tuned_22[lg_k - 1];
        }
        if (lg_N == 24) {
            return tuned_24[lg_k - 1];
        }
        if (lg_N == 26) {
            return tuned_26[lg_k - 1];
        }
    }
    if (lg_N == 26 && lg_k == 1) {
        return 16;
    }
    if (lg_N >= 24) {
        return 12;
    }
    if (lg_k == 1 || lg_k == 2) {
        return 2;
    }
    if (lg_k == 3) {
        return 4;
    }
    return 8;
}

int step1_threads_default() {
    const char* override = std::getenv("STEP1_THREADS");
    if (override != nullptr) {
        return std::max(1, std::atoi(override));
    }
    return 16;
}

struct NttScaleConfig {
    int lg_N;
    int lg_i;
    int ours_lg_k;
    int baseline_lg_k_begin;
    int baseline_lg_k_end;
    int reps;
    StageMapPolicy map_policy;
    const char* input_file;
};

NttScaleConfig ntt_scale_config(int lg_N) {
    switch (lg_N) {
    case 22:
        return {22, 12, 4, 1, 8, 3, StageMapPolicy::automatic, "../../input.bin"};
    case 24:
        return {24, 14, 4, 1, 8, 3, StageMapPolicy::automatic, "../../input-24.bin"};
    case 26:
        return {26, 16, 2, 1, 8, 3, StageMapPolicy::disabled, "../../input-26.bin"};
    default:
        std::cerr << "Unsupported lg_N = " << lg_N
                  << ". Supported NTT test scales are 22, 24, and 26." << std::endl;
        std::exit(1);
    }
}

int lg_i_for_domain(int lg_N) {
    return ntt_scale_config(lg_N).lg_i;
}

const char* input_file_for_domain(int lg_N) {
    return ntt_scale_config(lg_N).input_file;
}

size_t chunk_bytes_for(int lg_i, int lg_k) {
    return (size_t{1} << (lg_i + lg_k)) * sizeof(cpu::BLS12_381_Fr_G1);
}

std::string format_bytes(size_t bytes) {
    const double kb = static_cast<double>(bytes) / 1024.0;
    const double mb = kb / 1024.0;
    std::ostringstream out;
    const auto old_flags = out.flags();
    const auto old_precision = out.precision();
    if (mb >= 1.0) {
        out << std::fixed << std::setprecision(2) << mb << " MB";
    } else {
        out << std::fixed << std::setprecision(2) << kb << " KB";
    }
    out.flags(old_flags);
    out.precision(old_precision);
    return out.str();
}

bool should_build_stage_map(StageMapPolicy policy, int iterations) {
    if (iterations > 10 && policy != StageMapPolicy::enabled) {
        return false;
    }
    if (policy == StageMapPolicy::enabled) {
        return true;
    }
    return policy == StageMapPolicy::automatic;
}

void print_workspace() {
    std::cout << "[workspace] mode=lowmem, no full-GPU vector buffer" << std::endl;
}

bool compare_outputs(SyncedMemory expected, SyncedMemory actual) {
    const auto* expected_data = reinterpret_cast<const uint64_t*>(expected.mutable_cpu_data());
    const auto* actual_data = reinterpret_cast<const uint64_t*>(actual.mutable_cpu_data());
    const size_t field_words = 4;
    const size_t field_count = expected.size() / (field_words * sizeof(uint64_t));

    for (size_t i = 0; i < field_count; ++i) {
        const size_t word = i * field_words;
        if (std::memcmp(expected_data + word, actual_data + word, field_words * sizeof(uint64_t)) != 0) {
            std::cout << "mismatch at element " << i << "\nexpected:";
            for (size_t j = 0; j < field_words; ++j) {
                std::cout << " 0x" << std::hex << expected_data[word + j];
            }
            std::cout << "\nactual:  ";
            for (size_t j = 0; j < field_words; ++j) {
                std::cout << " 0x" << std::hex << actual_data[word + j];
            }
            std::cout << std::dec << std::endl;
            return false;
        }
    }

    std::cout << "outputs match for " << field_count << " elements" << std::endl;
    return true;
}

int run_check(int lg_k,
              int lg_N = 22,
              const char* file_path = nullptr,
              int lg_i_override = 0,
              StageMapPolicy map_policy = StageMapPolicy::automatic) {
    int lg_i = lg_i_override > 0 ? lg_i_override : lg_i_for_domain(lg_N);
    int lg_j = lg_N - lg_i;
    uint64_t N = uint64_t{1} << lg_N;
    int step1_thread = step1_threads_default();
    int step2_thread = step2_threads_for_lg_k(lg_k, lg_N);
    if (file_path == nullptr) {
        file_path = input_file_for_domain(lg_N);
    }

    Ntt ntt(32);
    SyncedMemory reference_input(N * 32);
    SyncedMemory ours_input(N * 32);
    read_file(file_path, reference_input.mutable_cpu_data());
    read_file(file_path, ours_input.mutable_cpu_data());

    SyncedMemory expected = ntt.forward(reference_input);
    cudaDeviceSynchronize();
    expected.mutable_cpu_data();

    print_workspace();

    StageIndexMap second_stage_map;
    StageIndexMap* second_stage_map_ptr = nullptr;
    const int second_iterations = lg_N - lg_i;
    if (should_build_stage_map(map_policy, second_iterations)) {
        const uint64_t chunk_elems = (uint64_t{1} << lg_i) * (uint64_t{1} << lg_k);
        const uint64_t chunk_count = N / chunk_elems;
        second_stage_map = build_stage_index_map(chunk_elems, chunk_count,
                                                 lg_i, second_iterations, step2_thread);
        second_stage_map_ptr = &second_stage_map;
    }

    our_ntt(ours_input, lg_N, lg_i, lg_j, lg_k,
            step1_thread, step2_thread, ntt, second_stage_map_ptr);
    cudaDeviceSynchronize();
    ours_input.mutable_cpu_data();

    return compare_outputs(expected, ours_input) ? 0 : 1;
}

int run_standard4_check(int lg_k,
                        int lg_N = 22,
                        const char* file_path = nullptr,
                        int lg_i_override = 0) {
    const int lg_i = lg_i_override > 0 ? lg_i_override : lg_i_for_domain(lg_N);
    const int lg_j = lg_N - lg_i;
    const uint64_t N = uint64_t{1} << lg_N;
    const int transpose_threads = step1_threads_default();
    if (file_path == nullptr) {
        file_path = input_file_for_domain(lg_N);
    }

    Ntt ntt(32);
    SyncedMemory reference_input(N * 32);
    SyncedMemory standard_input(N * 32);
    read_file(file_path, reference_input.mutable_cpu_data());
    read_file(file_path, standard_input.mutable_cpu_data());

    SyncedMemory expected = ntt.forward(reference_input);
    cudaDeviceSynchronize();
    expected.mutable_cpu_data();

    std::cout << "[workspace] mode=standard4, strided column H2D/D2H, contiguous row H2D/D2H"
              << std::endl;
    standard4_ntt(standard_input, lg_N, lg_i, lg_j, lg_k,
                  transpose_threads, ntt);
    cudaDeviceSynchronize();
    standard_input.mutable_cpu_data();

    return compare_outputs(expected, standard_input) ? 0 : 1;
}

int run_our_benchmark(int lg_N,
                      int lg_i,
                      int lg_k_begin,
                      int lg_k_end,
                      int reps,
                      const char* file_path,
                      StageMapPolicy map_policy = StageMapPolicy::automatic) {
    int lg_j = lg_N - lg_i;
    uint64_t N = uint64_t{1} << lg_N;
    int step1_thread = step1_threads_default();

    Ntt ntt(32);
    SyncedMemory input(N * 32);
    void* input_ptr = input.mutable_cpu_data();
    print_workspace();

    for (int lg_k = lg_k_begin; lg_k <= lg_k_end; ++lg_k) {
        int step2_thread = step2_threads_for_lg_k(lg_k, lg_N);
        const size_t chunk_bytes = chunk_bytes_for(lg_i, lg_k);
        StageIndexMap second_stage_map;
        StageIndexMap* second_stage_map_ptr = nullptr;
        const int second_iterations = lg_N - lg_i;
        if (should_build_stage_map(map_policy, second_iterations)) {
            const uint64_t chunk_elems = (uint64_t{1} << lg_i) * (uint64_t{1} << lg_k);
            const uint64_t chunk_count = N / chunk_elems;
            second_stage_map = build_stage_index_map(chunk_elems, chunk_count,
                                                     lg_i, second_iterations, step2_thread);
            second_stage_map_ptr = &second_stage_map;
        }
        std::cout << "=== lg_N = " << lg_N
                  << ", split = " << lg_i << "+" << lg_j
                  << ", mode = lowmem"
                  << ", chunk_size = " << format_bytes(chunk_bytes)
                  << ", step1_thread = " << step1_thread
                  << ", step2_thread = " << step2_thread
                  << " ===" << std::endl;

        for (int rep = 0; rep < reps; ++rep) {
            read_file(file_path, input_ptr);
            std::cout << "[our_ntt] run " << (rep + 1) << "/" << reps << std::endl;
            our_ntt(input, lg_N, lg_i, lg_j, lg_k,
                    step1_thread, step2_thread, ntt, second_stage_map_ptr);
        }
    }

    return 0;
}

int run_standard4_benchmark(int lg_N,
                            int lg_i,
                            int lg_k_begin,
                            int lg_k_end,
                            int reps,
                            const char* file_path) {
    const int lg_j = lg_N - lg_i;
    const uint64_t N = uint64_t{1} << lg_N;
    const int transpose_threads = step1_threads_default();

    Ntt ntt(32);
    SyncedMemory input(N * 32);
    void* input_ptr = input.mutable_cpu_data();

    std::cout << "[workspace] mode=standard4, strided column H2D/D2H, contiguous row H2D/D2H"
              << std::endl;
    for (int lg_k = lg_k_begin; lg_k <= lg_k_end; ++lg_k) {
        const size_t chunk_bytes = chunk_bytes_for(lg_i, lg_k);
        std::cout << "=== lg_N = " << lg_N
                  << ", split = " << lg_i << "+" << lg_j
                  << ", mode = baseline-standard4"
                  << ", chunk_size = " << format_bytes(chunk_bytes)
                  << ", transpose_threads = " << transpose_threads
                  << " ===" << std::endl;

        for (int rep = 0; rep < reps; ++rep) {
            read_file(file_path, input_ptr);
            std::cout << "[standard4_ntt] run " << (rep + 1) << "/" << reps << std::endl;
            standard4_ntt(input, lg_N, lg_i, lg_j, lg_k,
                          transpose_threads, ntt);
        }
    }

    return 0;
}

int run_configured_our_check(int lg_N) {
    const NttScaleConfig cfg = ntt_scale_config(lg_N);
    return run_check(cfg.ours_lg_k, cfg.lg_N, cfg.input_file, cfg.lg_i,
                     cfg.map_policy);
}

int run_configured_baseline_check(int lg_N) {
    const NttScaleConfig cfg = ntt_scale_config(lg_N);
    return run_standard4_check(cfg.ours_lg_k, cfg.lg_N, cfg.input_file, cfg.lg_i);
}

int run_configured_our_benchmark(int lg_N) {
    const NttScaleConfig cfg = ntt_scale_config(lg_N);
    return run_our_benchmark(cfg.lg_N, cfg.lg_i, cfg.ours_lg_k, cfg.ours_lg_k,
                             cfg.reps, cfg.input_file, cfg.map_policy);
}

int run_configured_baseline_benchmark(int lg_N) {
    const NttScaleConfig cfg = ntt_scale_config(lg_N);
    return run_standard4_benchmark(cfg.lg_N, cfg.lg_i,
                                   cfg.baseline_lg_k_begin,
                                   cfg.baseline_lg_k_end,
                                   cfg.reps, cfg.input_file);
}

void print_ntt_usage(const char* exe) {
    std::cerr << "Usage:\n"
              << "  " << exe << " --bench-split <lg_N>\n"
              << "  " << exe << " --bench-baseline <lg_N>\n"
              << "  " << exe << " --check-split <lg_N>\n"
              << "  " << exe << " --check-baseline <lg_N>\n"
              << "  " << exe << " --bench-sweep <lg_N> [reps]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_ntt_usage(argv[0]);
        return 1;
    }

    const std::string command(argv[1]);
    if (command == "--bench-baseline") {
        int lg_N = argc >= 3 ? std::atoi(argv[2]) : 22;
        return run_configured_baseline_benchmark(lg_N);
    }
    if (command == "--check-baseline") {
        int lg_N = argc >= 3 ? std::atoi(argv[2]) : 22;
        return run_configured_baseline_check(lg_N);
    }
    if (command == "--check-split") {
        int lg_N = argc >= 3 ? std::atoi(argv[2]) : 22;
        return run_configured_our_check(lg_N);
    }
    if (command == "--bench-split") {
        int lg_N = argc >= 3 ? std::atoi(argv[2]) : 22;
        return run_configured_our_benchmark(lg_N);
    }
    if (command == "--bench-sweep") {
        int lg_N = argc >= 3 ? std::atoi(argv[2]) : 22;
        int reps = argc >= 4 ? std::atoi(argv[3]) : 1;
        const NttScaleConfig cfg = ntt_scale_config(lg_N);
        return run_our_benchmark(cfg.lg_N, cfg.lg_i, 1, 8, reps,
                                 cfg.input_file, cfg.map_policy);
    }

    std::cerr << "Unknown command: " << command << "\n";
    print_ntt_usage(argv[0]);
    return 1;
}
