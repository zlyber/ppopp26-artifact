#include "structure.cuh"

uint64_t next_power_of_2(uint64_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;
    return x;
}

uint64_t total_size(CircuitC circuit){
    return std::max(circuit.n, circuit.lookup_len);
}

uint64_t circuit_bound(CircuitC circuit){
    return next_power_of_2(total_size(circuit));
}