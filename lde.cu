#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "caffe/interface.hpp"
#include "PLONK/utils/function.cuh"
#include "PLONK/utils/zkp/cuda/zksnark_ntt/ntt_kernel/ntt.cuh"

static_assert(sizeof(cpu::BLS12_381_Fr_G1) == 4 * sizeof(uint64_t),
              "LDE kernels copy field elements as four 64-bit limbs");

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

class HostBuffer {
public:
    HostBuffer() = default;

    explicit HostBuffer(size_t bytes, bool prefer_pinned = false) {
        allocate(bytes, prefer_pinned);
    }

    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    HostBuffer(HostBuffer&& other) noexcept {
        *this = std::move(other);
    }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            pinned_ = other.pinned_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
            other.pinned_ = false;
        }
        return *this;
    }

    ~HostBuffer() {
        release();
    }

    void allocate(size_t bytes, bool prefer_pinned = false) {
        release();
        bytes_ = bytes;
        if (bytes == 0) {
            return;
        }
        if (prefer_pinned) {
            void* p = nullptr;
            cudaError_t err = cudaMallocHost(&p, bytes);
            if (err == cudaSuccess) {
                ptr_ = p;
                pinned_ = true;
                return;
            }
            cudaGetLastError();
        }
        ptr_ = std::malloc(bytes);
        if (ptr_ == nullptr) {
            throw std::bad_alloc();
        }
        pinned_ = false;
    }

    void* data() {
        return ptr_;
    }

    const void* data() const {
        return ptr_;
    }

    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(ptr_);
    }

    size_t size() const {
        return bytes_;
    }

    bool pinned() const {
        return pinned_;
    }

private:
    void release() {
        if (ptr_ == nullptr) {
            return;
        }
        if (pinned_) {
            CUDA_CHECK(cudaFreeHost(ptr_));
        } else {
            std::free(ptr_);
        }
        ptr_ = nullptr;
        bytes_ = 0;
        pinned_ = false;
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    bool pinned_ = false;
};

void read_file_exact(const char* filename, void* data, size_t bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Error opening file: ") + filename);
    }
    file.read(reinterpret_cast<char*>(data), bytes);
    if (!file) {
        throw std::runtime_error(std::string("Error reading file: ") + filename);
    }
}

double measure_host_ms(const std::function<void()>& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::atoi(value) != 0;
}

uint64_t reverse_bits_host(uint64_t value, int bits) {
    uint64_t reversed = 0;
    for (int bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

std::vector<int> split_supported_stage(int iterations) {
    switch (iterations) {
        case 6:
        case 7:
        case 8:
        case 10:
            return {iterations};
        case 12:
            return {6, 6};
        case 13:
            return {6, 7};
        case 14:
            return {7, 7};
        case 15:
            return {7, 8};
        case 16:
            return {8, 8};
        case 17:
            return {7, 10};
        case 18:
            return {8, 10};
        default:
            throw std::runtime_error("unsupported stage split for CTkernel_no_rotate");
    }
}

struct LdePlan {
    int lg_domain_size = 0;
    int first_original_iterations = 0;
    int first_iterations = 0;
    std::vector<int> first_groups;
    std::vector<int> remaining_groups;
};

LdePlan make_lde_plan(int lg_N, int lambda) {
    LdePlan plan;
    plan.lg_domain_size = lg_N + lambda;
    plan.first_iterations = (plan.lg_domain_size == 29)
                                ? 17
                                : plan.lg_domain_size - 10;
    plan.first_original_iterations = plan.first_iterations - lambda;
    if (plan.first_original_iterations <= 0) {
        throw std::runtime_error("invalid LDE split");
    }
    plan.first_groups = split_supported_stage(plan.first_iterations);
    plan.remaining_groups =
        split_supported_stage(plan.lg_domain_size - plan.first_iterations);
    return plan;
}

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
void bit_rev_input_parallel(fr* dst,
                            const fr* src,
                            uint64_t count,
                            int lg_count,
                            int num_threads) {
    CpuThreadPool pool(num_threads);
    pool.parallel_for(count, [&](uint64_t begin, uint64_t end) {
        for (uint64_t i = begin; i < end; ++i) {
            dst[reverse_bits_host(i, lg_count)] = src[i];
        }
    });
}

template <typename fr>
void pad_lde_input_parallel(fr* dst,
                            const fr* src,
                            uint64_t count,
                            int lambda,
                            int num_threads) {
    CpuThreadPool pool(num_threads);
    pool.parallel_for(count, [&](uint64_t begin, uint64_t end) {
        for (uint64_t i = begin; i < end; ++i) {
            dst[i << lambda] = src[i];
        }
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
void gather_stage_input_chunk_range(fr* dst,
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
void gather_stage_input_chunk_parallel(fr* dst,
                                       const fr* src,
                                       uint64_t chunk,
                                       uint64_t chunk_elems,
                                       int stage,
                                       int iterations,
                                       int num_threads,
                                       CpuThreadPool* pool = nullptr) {
    const uint64_t threads_per_chunk = chunk_elems / 2;
    if (num_threads <= 1 || threads_per_chunk < 4096) {
        gather_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                       stage, iterations, 0, threads_per_chunk);
        return;
    }

    if (pool != nullptr && pool->worker_count() == num_threads) {
        pool->parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
            gather_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                           stage, iterations, begin, end);
        });
        return;
    }

    CpuThreadPool local_pool(num_threads);
    local_pool.parallel_for(threads_per_chunk, [&](uint64_t begin, uint64_t end) {
        gather_stage_input_chunk_range(dst, src, chunk, chunk_elems,
                                       stage, iterations, begin, end);
    });
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

        dst[rotate_stage_output_index(global0, stage, iterations)] = src[local0];
        dst[rotate_stage_output_index(global1, stage, iterations)] = src[local1];
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

__global__ void lde_spread_powers_kernel(cuda::fr* dst,
                                         const cuda::fr* src,
                                         uint64_t compact_elems,
                                         uint64_t compact_offset,
                                         int lambda,
                                         int lg_domain_size,
                                         const cuda::fr* partial_group_gen_powers) {
    const uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    const uint64_t expanded_elems = compact_elems << lambda;
    if (idx >= expanded_elems) {
        return;
    }

    cuda::fr value;
    value.zero();
    const uint64_t mask = (uint64_t{1} << lambda) - 1;
    if ((idx & mask) == 0) {
        const uint64_t compact_idx = idx >> lambda;
        value = src[compact_idx];
        const cuda::index_t pow =
            cuda::bit_rev(static_cast<cuda::index_t>(compact_offset + compact_idx),
                          lg_domain_size);
        value *= cuda::get_intermediate_root(pow, partial_group_gen_powers);
    }
    dst[idx] = value;
}

void launch_lde_spread_powers(void* dst,
                              const void* src,
                              uint64_t compact_elems,
                              uint64_t compact_offset,
                              int lambda,
                              int lg_domain_size,
                              Ntt& ntt,
                              cudaStream_t stream) {
    auto* params = reinterpret_cast<cuda::fr*>(ntt.Params.mutable_gpu_data());
    cuda::fr* partial_group_gen_powers = params + WINDOW_NUM * WINDOW_SIZE;
    const uint64_t expanded_elems = compact_elems << lambda;
    const int block = 256;
    const uint64_t grid = (expanded_elems + block - 1) / block;
    lde_spread_powers_kernel<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
        reinterpret_cast<cuda::fr*>(dst),
        reinterpret_cast<const cuda::fr*>(src),
        compact_elems,
        compact_offset,
        lambda,
        lg_domain_size,
        partial_group_gen_powers);
}

__device__ uint64_t reverse_bits_device_lde(uint64_t value, int bits) {
    uint64_t reversed = 0;
    for (int bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

__global__ void naive_block_bit_reverse_fields_kernel(uint64_t* data,
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
    const uint64_t rev = reverse_bits_device_lde(local, lg_block_elems);
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

void launch_naive_block_bit_reverse(void* data,
                                    uint64_t block_elems,
                                    uint64_t batch,
                                    int lg_block_elems,
                                    cudaStream_t stream) {
    const uint64_t total = block_elems * batch;
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    naive_block_bit_reverse_fields_kernel<<<static_cast<unsigned int>(grid_size),
                                             block_size, 0, stream>>>(
        reinterpret_cast<uint64_t*>(data),
        block_elems,
        batch,
        lg_block_elems);
}

__global__ void naive_transpose_fields_kernel(uint64_t* dst,
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

void launch_naive_transpose_fields(void* dst,
                                   const void* src,
                                   uint64_t rows,
                                   uint64_t cols,
                                   cudaStream_t stream) {
    const uint64_t total = rows * cols;
    const int block_size = 256;
    const uint64_t grid_size = (total + block_size - 1) / block_size;
    naive_transpose_fields_kernel<<<static_cast<unsigned int>(grid_size),
                                    block_size, 0, stream>>>(
        reinterpret_cast<uint64_t*>(dst),
        reinterpret_cast<const uint64_t*>(src),
        rows,
        cols);
}

__global__ void naive_standard4_twiddle_kernel(cuda::fr* data,
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

void launch_naive_standard4_twiddle(void* data,
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
    naive_standard4_twiddle_kernel<<<static_cast<unsigned int>(grid_size),
                                     block_size, 0, stream>>>(
        reinterpret_cast<cuda::fr*>(data),
        rows,
        cols,
        col_base,
        lg_N,
        partial_twiddles);
}

__global__ void naive_batched_ct_stage_kernel(cuda::fr* data,
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

void launch_naive_batched_ct_stage(void* data,
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
    naive_batched_ct_stage_kernel<<<static_cast<unsigned int>(grid_size),
                                    block_size, 0, stream>>>(
        reinterpret_cast<cuda::fr*>(data),
        len,
        batch,
        lg_len,
        stage,
        partial_twiddles);
}

void run_naive_batched_simple_ntt(void* data,
                                  uint64_t batch,
                                  uint64_t len,
                                  int lg_len,
                                  Ntt& ntt,
                                  cudaStream_t stream) {
    launch_naive_block_bit_reverse(data, len, batch, lg_len, stream);
    for (int stage = 0; stage < lg_len; ++stage) {
        launch_naive_batched_ct_stage(data, batch, len, lg_len, stage, ntt, stream);
    }
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
    const uint64_t grid = (threads + block - 1) / block;
    scatter_stage_kernel<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
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
    const uint64_t grid = (threads + block - 1) / block;
    transition_stage_kernel<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
        reinterpret_cast<uint64_t*>(dst),
        reinterpret_cast<const uint64_t*>(src),
        chunk,
        chunk_elems,
        prev_stage,
        prev_iterations,
        next_stage,
        next_iterations);
}

void run_first_lde_pass(cpu::BLS12_381_Fr_G1* bitrev_input,
                        cpu::BLS12_381_Fr_G1* stage_out,
                        uint64_t compact_chunk_elems,
                        uint64_t chunk_count,
                        int lambda,
                        int first_iterations,
                        int lg_domain_size,
                        Ntt& ntt) {
    const int pipeline_slots = 2;
    const uint64_t expanded_chunk_elems = compact_chunk_elems << lambda;
    const size_t compact_chunk_bytes = compact_chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);
    const size_t expanded_chunk_bytes = expanded_chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);

    SyncedMemory d_compact[pipeline_slots] = {
        SyncedMemory(compact_chunk_bytes),
        SyncedMemory(compact_chunk_bytes)
    };
    SyncedMemory d_work[pipeline_slots] = {
        SyncedMemory(expanded_chunk_bytes),
        SyncedMemory(expanded_chunk_bytes)
    };
    SyncedMemory d_scratch[pipeline_slots] = {
        SyncedMemory(expanded_chunk_bytes),
        SyncedMemory(expanded_chunk_bytes)
    };

    void* d_compact_ptr[pipeline_slots];
    void* d_work_ptr[pipeline_slots];
    void* d_scratch_ptr[pipeline_slots];
    cudaStream_t streams[pipeline_slots];
    cudaEvent_t done[pipeline_slots];

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        d_compact_ptr[slot] = d_compact[slot].mutable_gpu_data();
        d_work_ptr[slot] = d_work[slot].mutable_gpu_data();
        d_scratch_ptr[slot] = d_scratch[slot].mutable_gpu_data();
        cudaStreamCreate(&streams[slot]);
        cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming);
    }

    for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int slot = chunk % pipeline_slots;
        if (chunk >= static_cast<uint64_t>(pipeline_slots)) {
            cudaEventSynchronize(done[slot]);
        }

        cudaMemcpyAsync(d_compact_ptr[slot],
                        bitrev_input + chunk * compact_chunk_elems,
                        compact_chunk_bytes,
                        cudaMemcpyHostToDevice,
                        streams[slot]);
        launch_lde_spread_powers(d_work_ptr[slot],
                                 d_compact_ptr[slot],
                                 compact_chunk_elems,
                                 chunk * compact_chunk_elems,
                                 lambda,
                                 lg_domain_size,
                                 ntt,
                                 streams[slot]);

        const std::vector<int> first_groups = split_supported_stage(first_iterations);
        if (first_groups.size() == 1) {
            ntt.stage_no_rotate(d_work[slot], lg_domain_size,
                                first_groups[0], 0, chunk, streams[slot]);
            launch_scatter_stage(d_scratch_ptr[slot], d_work_ptr[slot],
                                 chunk, expanded_chunk_elems,
                                 0, first_groups[0], streams[slot]);
            cudaMemcpyAsync(stage_out + chunk * expanded_chunk_elems,
                            d_scratch_ptr[slot],
                            expanded_chunk_bytes,
                            cudaMemcpyDeviceToHost,
                            streams[slot]);
        } else {
            assert(first_groups.size() == 2);
            const int first_step = first_groups[0];
            const int second_step = first_groups[1];
            ntt.stage_no_rotate(d_work[slot], lg_domain_size,
                                first_step, 0, chunk, streams[slot]);
            launch_transition_stage(d_scratch_ptr[slot], d_work_ptr[slot],
                                    chunk, expanded_chunk_elems,
                                    0, first_step,
                                    first_step, second_step,
                                    streams[slot]);
            ntt.stage_no_rotate(d_scratch[slot], lg_domain_size,
                                second_step, first_step, chunk, streams[slot]);
            launch_scatter_stage(d_work_ptr[slot], d_scratch_ptr[slot],
                                 chunk, expanded_chunk_elems,
                                 first_step, second_step, streams[slot]);
            cudaMemcpyAsync(stage_out + chunk * expanded_chunk_elems,
                            d_work_ptr[slot],
                            expanded_chunk_bytes,
                            cudaMemcpyDeviceToHost,
                            streams[slot]);
        }
        cudaEventRecord(done[slot], streams[slot]);
    }

    for (int slot = 0; slot < pipeline_slots; ++slot) {
        cudaEventSynchronize(done[slot]);
        cudaEventDestroy(done[slot]);
        cudaStreamDestroy(streams[slot]);
    }
}

void run_lde_stage_pass(cpu::BLS12_381_Fr_G1* stage_in,
                        cpu::BLS12_381_Fr_G1* output,
                        uint64_t expanded_chunk_elems,
                        uint64_t chunk_count,
                        int stage,
                        int iterations,
                        int lg_domain_size,
                        int num_threads,
                        Ntt& ntt,
                        const StageIndexMap* stage_map = nullptr) {
    assert(iterations <= 10);
    const int pipeline_slots = 2;
    const size_t chunk_bytes = expanded_chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);

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
                         stage_map->matches(expanded_chunk_elems, chunk_count, stage, iterations);

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
                                                    chunk,
                                                    expanded_chunk_elems,
                                                    stage,
                                                    iterations,
                                                    num_threads,
                                                    scatter_pools[slot]);
            }
        });
        scatter_pending[slot] = true;
    };

    auto gather_current = [&](int slot, uint64_t chunk) {
        if (use_map) {
            gather_stage_input_map_parallel(h_in_ptrs[slot], stage_in,
                                            *stage_map, chunk,
                                            num_threads, &gather_pool);
        } else {
            gather_stage_input_chunk_parallel(h_in_ptrs[slot], stage_in,
                                              chunk, expanded_chunk_elems,
                                              stage, iterations,
                                              num_threads, &gather_pool);
        }
    };

    for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int slot = chunk % pipeline_slots;
        if (slot_valid[slot]) {
            cudaEventSynchronize(done[slot]);
            launch_scatter(slot, completed_chunk[slot]);
            slot_valid[slot] = false;
        }

        gather_current(slot, chunk);
        cudaMemcpyAsync(d_ptrs[slot], h_in_ptrs[slot], chunk_bytes,
                        cudaMemcpyHostToDevice, streams[slot]);
        ntt.stage_no_rotate(d_buffers[slot], lg_domain_size,
                            iterations, stage,
                            chunk, streams[slot]);
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

float our_lde(cpu::BLS12_381_Fr_G1* input,
              int lg_N,
              int lambda,
              int lg_k,
              int step1_thread,
              int step2_thread,
              Ntt& ntt) {
    const LdePlan plan = make_lde_plan(lg_N, lambda);
    if (plan.first_original_iterations + lg_k > lg_N) {
        throw std::runtime_error("lg_k is too large for the selected LDE split");
    }

    const uint64_t N = uint64_t{1} << lg_N;
    const uint64_t expanded_N = uint64_t{1} << plan.lg_domain_size;
    const uint64_t k = uint64_t{1} << lg_k;
    const uint64_t compact_chunk_elems = (uint64_t{1} << plan.first_original_iterations) * k;
    const uint64_t expanded_chunk_elems = compact_chunk_elems << lambda;
    const uint64_t chunk_count = N / compact_chunk_elems;
    const size_t field_bytes = sizeof(cpu::BLS12_381_Fr_G1);

    HostBuffer bitrev_input(N * field_bytes, true);
    HostBuffer stage1_out(expanded_N * field_bytes, true);
    HostBuffer output(expanded_N * field_bytes, false);
    auto* bitrev_input_cpu = bitrev_input.as<cpu::BLS12_381_Fr_G1>();
    auto* stage1_out_cpu = stage1_out.as<cpu::BLS12_381_Fr_G1>();
    auto* output_cpu = output.as<cpu::BLS12_381_Fr_G1>();

    std::vector<StageIndexMap> stage_maps;
    const char* stage_map_env = std::getenv("LDE_STAGE_MAP");
    const bool use_stage_map = stage_map_env == nullptr || std::atoi(stage_map_env) != 0;
    if (use_stage_map) {
        int map_stage = plan.first_iterations;
        for (int iterations : plan.remaining_groups) {
            stage_maps.push_back(build_stage_index_map(expanded_chunk_elems, chunk_count,
                                                       map_stage, iterations, step2_thread));
            map_stage += iterations;
        }
    }
    std::memset(output_cpu, 0, expanded_N * field_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    double bitrev_ms = 0.0;
    double first_ms = 0.0;
    std::vector<double> stage_ms;
    const bool profile = env_enabled("LDE_PROFILE");

    bitrev_ms = measure_host_ms([&]() {
        bit_rev_input_parallel(bitrev_input_cpu, input, N, lg_N, step1_thread);
    });
    first_ms = measure_host_ms([&]() {
        run_first_lde_pass(bitrev_input_cpu, stage1_out_cpu,
                           compact_chunk_elems, chunk_count,
                           lambda, plan.first_iterations, plan.lg_domain_size, ntt);
    });

    cpu::BLS12_381_Fr_G1* current = stage1_out_cpu;
    cpu::BLS12_381_Fr_G1* next = output_cpu;
    int stage = plan.first_iterations;
    size_t map_idx = 0;
    for (int iterations : plan.remaining_groups) {
        const StageIndexMap* stage_map =
            map_idx < stage_maps.size() ? &stage_maps[map_idx] : nullptr;
        const double pass_ms = measure_host_ms([&]() {
            run_lde_stage_pass(current, next,
                               expanded_chunk_elems, chunk_count,
                               stage, iterations,
                               plan.lg_domain_size, step2_thread, ntt,
                               stage_map);
        });
        stage_ms.push_back(pass_ms);
        stage += iterations;
        ++map_idx;
        std::swap(current, next);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "our-lde execution time: " << milliseconds << " ms" << std::endl;
    if (profile) {
        std::cout << "[lde-profile] bitrev=" << bitrev_ms
                  << " ms, first_pass=" << first_ms;
        for (size_t i = 0; i < stage_ms.size(); ++i) {
            std::cout << " ms, stage_pass" << i << "=" << stage_ms[i];
        }
        std::cout << " ms" << std::endl;
    }
    return milliseconds;
}

float naive_lde(cpu::BLS12_381_Fr_G1* input,
                int lg_N,
                int lambda,
                int lg_k,
                int pad_threads,
                int transpose_threads,
                Ntt& ntt) {
    const int lg_domain_size = lg_N + lambda;
    const int lg_j = 10;
    const int lg_i = lg_domain_size - lg_j;
    if (lg_i <= 0) {
        throw std::runtime_error("invalid naive LDE split");
    }
    if (lg_k > lg_i || lg_k > lg_j) {
        throw std::runtime_error("lg_k is too large for naive LDE split");
    }

    const uint64_t N = uint64_t{1} << lg_N;
    const uint64_t expanded_N = uint64_t{1} << lg_domain_size;
    const uint64_t I = uint64_t{1} << lg_i;
    const uint64_t J = uint64_t{1} << lg_j;
    const uint64_t k = uint64_t{1} << lg_k;
    const size_t field_bytes = sizeof(cpu::BLS12_381_Fr_G1);
    const uint64_t first_chunk_elems = I * k;
    const uint64_t second_chunk_elems = J * k;
    const size_t expanded_bytes = expanded_N * field_bytes;

    HostBuffer expanded(expanded_bytes, true);
    HostBuffer transposed(expanded_bytes, false);
    auto* expanded_cpu = expanded.as<cpu::BLS12_381_Fr_G1>();
    auto* transposed_cpu = transposed.as<cpu::BLS12_381_Fr_G1>();

    SyncedMemory d_col_tile(first_chunk_elems * field_bytes);
    SyncedMemory d_col_batch(first_chunk_elems * field_bytes);
    SyncedMemory d_row_tile(second_chunk_elems * field_bytes);
    void* d_col_tile_ptr = d_col_tile.mutable_gpu_data();
    void* d_col_batch_ptr = d_col_batch.mutable_gpu_data();
    void* d_row_tile_ptr = d_row_tile.mutable_gpu_data();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    double pad_ms = 0.0;
    double column_ms = 0.0;
    double row_ms = 0.0;
    double transpose_ms = 0.0;
    const bool profile = env_enabled("LDE_PROFILE");

    pad_ms = measure_host_ms([&]() {
        std::memset(expanded_cpu, 0, expanded_bytes);
        pad_lde_input_parallel(expanded_cpu, input, N, lambda, pad_threads);
    });

    column_ms = measure_host_ms([&]() {
        for (uint64_t col_base = 0; col_base < J; col_base += k) {
            cudaMemcpy2DAsync(d_col_tile_ptr,
                              k * field_bytes,
                              expanded_cpu + col_base,
                              J * field_bytes,
                              k * field_bytes,
                              I,
                              cudaMemcpyHostToDevice,
                              stream);
            launch_naive_transpose_fields(d_col_batch_ptr, d_col_tile_ptr, I, k, stream);
            run_naive_batched_simple_ntt(d_col_batch_ptr, k, I, lg_i, ntt, stream);
            launch_naive_standard4_twiddle(d_col_batch_ptr, I, k, col_base,
                                           lg_domain_size, ntt, stream);
            launch_naive_transpose_fields(d_col_tile_ptr, d_col_batch_ptr, k, I, stream);
            cudaMemcpy2DAsync(expanded_cpu + col_base,
                              J * field_bytes,
                              d_col_tile_ptr,
                              k * field_bytes,
                              k * field_bytes,
                              I,
                              cudaMemcpyDeviceToHost,
                              stream);
            cudaStreamSynchronize(stream);
        }
    });

    row_ms = measure_host_ms([&]() {
        const size_t row_chunk_bytes = second_chunk_elems * field_bytes;
        for (uint64_t row_base = 0; row_base < I; row_base += k) {
            auto* row_ptr = expanded_cpu + row_base * J;
            cudaMemcpyAsync(d_row_tile_ptr, row_ptr, row_chunk_bytes,
                            cudaMemcpyHostToDevice, stream);
            run_naive_batched_simple_ntt(d_row_tile_ptr, k, J, lg_j, ntt, stream);
            cudaMemcpyAsync(row_ptr, d_row_tile_ptr, row_chunk_bytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    });

    transpose_ms = measure_host_ms([&]() {
        transpose_matrix_parallel(expanded_cpu, transposed_cpu, I, J, transpose_threads);
        std::memcpy(expanded_cpu, transposed_cpu, expanded_bytes);
    });

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    std::cout << "naive-lde execution time: " << milliseconds << " ms" << std::endl;
    if (profile) {
        std::cout << "[naive-lde-profile] pad=" << pad_ms
                  << " ms, column_ntt=" << column_ms
                  << " ms, row_ntt=" << row_ms
                  << " ms, final_transpose=" << transpose_ms
                  << " ms" << std::endl;
    }
    return milliseconds;
}

int step1_threads_default() {
    const char* override = std::getenv("STEP1_THREADS");
    if (override != nullptr) {
        return std::max(1, std::atoi(override));
    }
    return 16;
}

int step2_threads_default(int lg_domain_size) {
    const char* override = std::getenv("STEP2_THREADS");
    if (override != nullptr) {
        return std::max(1, std::atoi(override));
    }
    return lg_domain_size >= 24 ? 12 : 8;
}

int our_lde_step2_threads_for_lg_k(int lg_N, int lambda, int lg_k) {
    const char* override = std::getenv("STEP2_THREADS");
    if (override != nullptr) {
        return std::max(1, std::atoi(override));
    }
    if (lg_k >= 1 && lg_k <= 8) {
        static const int tuned_20_l2[8] = {8, 4, 12, 8, 12, 12, 4, 12};
        static const int tuned_20_l3[8] = {4, 4, 8, 12, 12, 12, 12, 12};
        static const int tuned_22_l2[8] = {4, 8, 12, 12, 12, 12, 12, 12};
        static const int tuned_22_l3[8] = {8, 12, 12, 12, 12, 12, 12, 12};
        static const int tuned_24_l2[8] = {12, 12, 12, 12, 12, 12, 12, 12};
        static const int tuned_24_l3[8] = {12, 12, 12, 12, 12, 12, 12, 8};
        if (lg_N == 20 && lambda == 2) {
            return tuned_20_l2[lg_k - 1];
        }
        if (lg_N == 20 && lambda == 3) {
            return tuned_20_l3[lg_k - 1];
        }
        if (lg_N == 22 && lambda == 2) {
            return tuned_22_l2[lg_k - 1];
        }
        if (lg_N == 22 && lambda == 3) {
            return tuned_22_l3[lg_k - 1];
        }
        if (lg_N == 24 && lambda == 2) {
            return tuned_24_l2[lg_k - 1];
        }
        if (lg_N == 24 && lambda == 3) {
            return tuned_24_l3[lg_k - 1];
        }
    }
    return step2_threads_default(lg_N + lambda);
}

struct LdeScaleConfig {
    int lg_N;
    int lambda;
    int ours_lg_k;
    int reps;
    const char* input_file;
};

LdeScaleConfig lde_scale_config(int lg_N, int lambda) {
    switch (lg_N) {
    case 20:
        if (lambda == 2) {
            return {20, 2, 4, 3, "../../input.bin"};
        }
        if (lambda == 3) {
            return {20, 3, 4, 3, "../../input.bin"};
        }
        break;
    case 22:
        if (lambda == 2) {
            return {22, 2, 3, 3, "../../input.bin"};
        }
        if (lambda == 3) {
            return {22, 3, 2, 3, "../../input.bin"};
        }
        break;
    case 24:
        if (lambda == 2) {
            return {24, 2, 1, 3, "../../input-24.bin"};
        }
        if (lambda == 3) {
            return {24, 3, 1, 3, "../../input-24.bin"};
        }
        break;
    default:
        break;
    }
    throw std::runtime_error("unsupported LDE test config; supported lg_N={20,22,24}, lambda={2,3}");
}

std::string format_bytes(size_t bytes) {
    const double kb = static_cast<double>(bytes) / 1024.0;
    const double mb = kb / 1024.0;
    std::ostringstream out;
    if (mb >= 1.0) {
        out << std::fixed << std::setprecision(2) << mb << " MB";
    } else {
        out << std::fixed << std::setprecision(2) << kb << " KB";
    }
    return out.str();
}

size_t lde_compact_chunk_bytes(int lg_N, int lambda, int lg_k) {
    const LdePlan plan = make_lde_plan(lg_N, lambda);
    const uint64_t compact_chunk_elems =
        (uint64_t{1} << plan.first_original_iterations) * (uint64_t{1} << lg_k);
    return compact_chunk_elems * sizeof(cpu::BLS12_381_Fr_G1);
}

size_t lde_expanded_chunk_bytes(int lg_N, int lambda, int lg_k) {
    return lde_compact_chunk_bytes(lg_N, lambda, lg_k) << lambda;
}

size_t naive_col_chunk_bytes(int lg_N, int lambda, int lg_k) {
    const int lg_domain_size = lg_N + lambda;
    const int lg_j = 10;
    const int lg_i = lg_domain_size - lg_j;
    return (size_t{1} << (lg_i + lg_k)) * sizeof(cpu::BLS12_381_Fr_G1);
}

size_t naive_row_chunk_bytes(int lg_k) {
    const int lg_j = 10;
    return (size_t{1} << (lg_j + lg_k)) * sizeof(cpu::BLS12_381_Fr_G1);
}

void print_workspace(int lg_N, int lambda, int lg_k) {
    const LdePlan plan = make_lde_plan(lg_N, lambda);
    const uint64_t compact_chunk_elems =
        (uint64_t{1} << plan.first_original_iterations) * (uint64_t{1} << lg_k);
    const uint64_t expanded_chunk_elems = compact_chunk_elems << lambda;
    const double mib = 1024.0 * 1024.0;
    const size_t field_bytes = sizeof(cpu::BLS12_381_Fr_G1);
    const double compact_mb = compact_chunk_elems * field_bytes / mib;
    const double expanded_mb = expanded_chunk_elems * field_bytes / mib;
    const double full_mb = (uint64_t{1} << (lg_N + lambda)) * field_bytes / mib;
    std::cout << std::fixed << std::setprecision(2)
              << "[workspace] compact_chunk=" << compact_mb
              << " MiB, expanded_chunk=" << expanded_mb
              << " MiB, full_expanded_host_buffer=" << full_mb
              << " MiB, split=" << plan.first_iterations
              << "+" << (plan.lg_domain_size - plan.first_iterations)
              << std::endl;
}

void print_naive_workspace(int lg_N, int lambda, int lg_k) {
    const int lg_domain_size = lg_N + lambda;
    const int lg_j = 10;
    const int lg_i = lg_domain_size - lg_j;
    const uint64_t I = uint64_t{1} << lg_i;
    const uint64_t J = uint64_t{1} << lg_j;
    const uint64_t k = uint64_t{1} << lg_k;
    const double mib = 1024.0 * 1024.0;
    const size_t field_bytes = sizeof(cpu::BLS12_381_Fr_G1);
    const double col_chunk_mb = I * k * field_bytes / mib;
    const double row_chunk_mb = J * k * field_bytes / mib;
    const double full_mb = (uint64_t{1} << lg_domain_size) * field_bytes / mib;
    std::cout << std::fixed << std::setprecision(2)
              << "[workspace] mode=naive, full_expanded_host_buffer=" << full_mb
              << " MiB, col_chunk=" << col_chunk_mb
              << " MiB, row_chunk=" << row_chunk_mb
              << " MiB, split=" << lg_i
              << "+" << lg_j
              << std::endl;
}

int run_benchmark(int lg_N,
                  int lambda,
                  int lg_k_begin,
                  int lg_k_end,
                  int reps,
                  const char* file_path) {
    const uint64_t N = uint64_t{1} << lg_N;
    const size_t input_bytes = N * sizeof(cpu::BLS12_381_Fr_G1);
    HostBuffer input(input_bytes, false);
    auto* input_cpu = input.as<cpu::BLS12_381_Fr_G1>();
    read_file_exact(file_path, input_cpu, input_bytes);

    const int step1_thread = step1_threads_default();
    Ntt ntt(32);

    std::cout << "[lde] lg_N=" << lg_N
              << ", lambda=" << lambda
              << ", input=" << file_path
              << ", reps=" << reps
              << ", step1_thread=" << step1_thread
              << std::endl;

    for (int lg_k = lg_k_begin; lg_k <= lg_k_end; ++lg_k) {
        const int step2_thread = our_lde_step2_threads_for_lg_k(lg_N, lambda, lg_k);
        std::cout << "=== lde lg_N = " << lg_N
                  << ", lambda = " << lambda
                  << ", compact_chunk = "
                  << format_bytes(lde_compact_chunk_bytes(lg_N, lambda, lg_k))
                  << ", expanded_chunk = "
                  << format_bytes(lde_expanded_chunk_bytes(lg_N, lambda, lg_k))
                  << ", step2_thread = " << step2_thread
                  << " ===" << std::endl;
        print_workspace(lg_N, lambda, lg_k);

        double sum = 0.0;
        for (int rep = 0; rep < reps; ++rep) {
            std::cout << "[our_lde] run " << (rep + 1) << "/" << reps << std::endl;
            sum += our_lde(input_cpu, lg_N, lambda, lg_k,
                           step1_thread, step2_thread, ntt);
        }
        std::cout << "[our_lde] avg lg_N=" << lg_N
                  << ", lambda=" << lambda
                  << ", compact_chunk="
                  << format_bytes(lde_compact_chunk_bytes(lg_N, lambda, lg_k))
                  << ", expanded_chunk="
                  << format_bytes(lde_expanded_chunk_bytes(lg_N, lambda, lg_k))
                  << ": " << (sum / reps) << " ms" << std::endl;
    }

    return 0;
}

int run_naive_benchmark(int lg_N,
                        int lambda,
                        int lg_k_begin,
                        int lg_k_end,
                        int reps,
                        const char* file_path) {
    const uint64_t N = uint64_t{1} << lg_N;
    const size_t input_bytes = N * sizeof(cpu::BLS12_381_Fr_G1);
    HostBuffer input(input_bytes, false);
    auto* input_cpu = input.as<cpu::BLS12_381_Fr_G1>();
    read_file_exact(file_path, input_cpu, input_bytes);

    const int pad_thread = step1_threads_default();
    const int transpose_thread = step2_threads_default(lg_N + lambda);
    Ntt ntt(32);

    std::cout << "[naive-lde] lg_N=" << lg_N
              << ", lambda=" << lambda
              << ", input=" << file_path
              << ", reps=" << reps
              << ", pad_thread=" << pad_thread
              << ", transpose_thread=" << transpose_thread
              << std::endl;

    for (int lg_k = lg_k_begin; lg_k <= lg_k_end; ++lg_k) {
        std::cout << "=== naive lde lg_N = " << lg_N
                  << ", lambda = " << lambda
                  << ", col_chunk = "
                  << format_bytes(naive_col_chunk_bytes(lg_N, lambda, lg_k))
                  << ", row_chunk = "
                  << format_bytes(naive_row_chunk_bytes(lg_k))
                  << " ===" << std::endl;
        print_naive_workspace(lg_N, lambda, lg_k);

        double sum = 0.0;
        for (int rep = 0; rep < reps; ++rep) {
            std::cout << "[naive_lde] run " << (rep + 1) << "/" << reps << std::endl;
            sum += naive_lde(input_cpu, lg_N, lambda, lg_k,
                             pad_thread, transpose_thread, ntt);
        }
        std::cout << "[naive_lde] avg lg_N=" << lg_N
                  << ", lambda=" << lambda
                  << ", col_chunk="
                  << format_bytes(naive_col_chunk_bytes(lg_N, lambda, lg_k))
                  << ", row_chunk="
                  << format_bytes(naive_row_chunk_bytes(lg_k))
                  << ": " << (sum / reps) << " ms" << std::endl;
    }

    return 0;
}

int run_configured_benchmark(int lg_N, int lambda) {
    const LdeScaleConfig cfg = lde_scale_config(lg_N, lambda);
    return run_benchmark(cfg.lg_N, cfg.lambda, cfg.ours_lg_k, cfg.ours_lg_k,
                         cfg.reps, cfg.input_file);
}

int run_configured_naive_benchmark(int lg_N, int lambda) {
    const LdeScaleConfig cfg = lde_scale_config(lg_N, lambda);
    return run_naive_benchmark(cfg.lg_N, cfg.lambda, 1, 8,
                               cfg.reps, cfg.input_file);
}

int run_configured_scale_benchmark(int lg_N) {
    for (int lambda : {2, 3}) {
        int rc = run_configured_benchmark(lg_N, lambda);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

int run_configured_scale_naive_benchmark(int lg_N) {
    for (int lambda : {2, 3}) {
        int rc = run_configured_naive_benchmark(lg_N, lambda);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

int run_sweep_benchmark(int lg_N, int lambda, int reps) {
    const LdeScaleConfig cfg = lde_scale_config(lg_N, lambda);
    return run_benchmark(cfg.lg_N, cfg.lambda, 1, 8, reps, cfg.input_file);
}

int run_default_suite() {
    for (int lg_N : {20, 22, 24}) {
        int rc = run_configured_scale_benchmark(lg_N);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            return run_default_suite();
        }
        if (std::strcmp(argv[1], "--bench") == 0) {
            if (argc < 3 || argc > 4) {
                std::cerr << "Usage: " << argv[0]
                          << " --bench <lg_N> [lambda]\n";
                return 1;
            }
            const int lg_N = std::atoi(argv[2]);
            if (argc == 4) {
                return run_configured_benchmark(lg_N, std::atoi(argv[3]));
            }
            return run_configured_scale_benchmark(lg_N);
        }
        if (std::strcmp(argv[1], "--bench-sweep") == 0) {
            if (argc < 4 || argc > 5) {
                std::cerr << "Usage: " << argv[0]
                          << " --bench-sweep <lg_N> <lambda> [reps]\n";
                return 1;
            }
            const int lg_N = std::atoi(argv[2]);
            const int lambda = std::atoi(argv[3]);
            const int reps = argc == 5 ? std::atoi(argv[4]) : 1;
            return run_sweep_benchmark(lg_N, lambda, reps);
        }
        if (std::strcmp(argv[1], "--bench-naive") == 0) {
            if (argc < 3 || argc > 4) {
                std::cerr << "Usage: " << argv[0]
                          << " --bench-naive <lg_N> [lambda]\n";
                return 1;
            }
            const int lg_N = std::atoi(argv[2]);
            if (argc == 4) {
                return run_configured_naive_benchmark(lg_N, std::atoi(argv[3]));
            }
            return run_configured_scale_naive_benchmark(lg_N);
        }
        std::cerr << "Unknown command: " << argv[1] << "\n";
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
}
