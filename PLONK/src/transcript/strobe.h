#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <vector>

constexpr int STROBE_R = 166;

constexpr int FLAG_I = 1;
constexpr int FLAG_A = 1 << 1;
constexpr int FLAG_C = 1 << 2;
constexpr int FLAG_T = 1 << 3;
constexpr int FLAG_M = 1 << 4;
constexpr int FLAG_K = 1 << 5;

constexpr int KECCAK_F_ROUND_COUNT = 24;

constexpr int PLEN = 25;

constexpr uint8_t RHO[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};

constexpr uint8_t PI[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

constexpr uint64_t RC[24] = {
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008
};

void transmute_state(const uint8_t* st, uint64_t* u64_values);

void transmute_inverse(const uint64_t* st, uint8_t* u8_array);

uint64_t rotate_left(uint64_t value, int n);

void keccak_p(uint64_t* state, int round_count);



class Strobe128 {
public:
    uint8_t state[200];
    int pos;
    int pos_begin;
    int cur_flags;

    static Strobe128 new_instance(std::string protocol_label);

    void run_f();

    void absorb(std::vector<uint8_t>& data);

    void squeeze(std::vector<uint8_t>& data);

    void begin_op(int flags, bool more);

    void meta_ad(std::vector<uint8_t>& data, bool more);

    void ad(std::vector<uint8_t>& data, bool more);

    void prf(std::vector<uint8_t>& data, bool more);
};

std::vector<uint8_t> str_to_u8(const std::string str);