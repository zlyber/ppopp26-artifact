#include "PLONK/src/transcript/strobe.h"



void transmute_state(const uint8_t* st, uint64_t* u64_values) {
    for (size_t i = 0; i < 200; i += 8) {
        memcpy(&u64_values[i / 8], &st[i], sizeof(uint64_t));
    }
}

void transmute_inverse(const uint64_t* st, uint8_t* u8_array) {
    for (size_t i = 0; i < 25; ++i) {
        memcpy(&u8_array[i * 8], &st[i], sizeof(uint64_t));
    }
}

uint64_t rotate_left(uint64_t value, int n) {
    return ((value << n) | (value >> (64 - n))) & 0xFFFFFFFFFFFFFFFF;
}

void keccak_p(uint64_t* state, int round_count) {
    if (round_count > KECCAK_F_ROUND_COUNT) {
        throw std::invalid_argument("A round_count greater than KECCAK_F_ROUND_COUNT is not supported!");
    }

    const uint64_t* round_consts = &RC[KECCAK_F_ROUND_COUNT - round_count];

    for (int i = 0; i < round_count; ++i) {
        uint64_t rc = round_consts[i];
        uint64_t array[5] = {0};

        // Theta
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                array[x] ^= state[5 * y + x];
            }
        }

        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                uint64_t t1 = array[(x + 4) % 5];
                uint64_t t2 = rotate_left(array[(x + 1) % 5], 1);
                state[5 * y + x] ^= t1 ^ t2;
            }
        }

        // Rho and Pi
        uint64_t last = state[1];
        for (int x = 0; x < 24; ++x) {
            array[0] = state[PI[x]];
            state[PI[x]] = rotate_left(last, RHO[x]);
            last = array[0];
        }

        // Chi
        for (int y_step = 0; y_step < 5; ++y_step) {
            int y = 5 * y_step;
            for (int x = 0; x < 5; ++x) {
                array[x] = state[y + x];
            }

            for (int x = 0; x < 5; ++x) {
                uint64_t t1 = ~array[(x + 1) % 5];
                uint64_t t2 = array[(x + 2) % 5];
                state[y + x] = array[x] ^ (t1 & t2);
            }
        }

        // Iota
        state[0] ^= rc;
    }
}


Strobe128 Strobe128::new_instance(const char* protocol_label) {
    Strobe128 strobe;
    memset(strobe.state, 0, 200);
    strobe.state[0] = 1;
    strobe.state[1] = STROBE_R + 2;
    strobe.state[2] = 1;
    strobe.state[3] = 0;
    strobe.state[4] = 1;
    strobe.state[5] = 96;
    memcpy(&strobe.state[6], "STROBEv1.0.2", 12);

    uint64_t st_u64[25];
    transmute_state(strobe.state, st_u64);
    keccak_p(st_u64, KECCAK_F_ROUND_COUNT);
    transmute_inverse(st_u64, strobe.state);

    strobe.pos = 0;
    strobe.pos_begin = 0;
    strobe.cur_flags = 0;
    strobe.meta_ad(protocol_label, false);

    return strobe;
}

void Strobe128::run_f() {
    state[pos] ^= pos_begin;
    state[pos + 1] ^= 0x04;
    state[STROBE_R + 1] ^= 0x80;

    uint64_t st_u64[25];
    transmute_state(state, st_u64);
    keccak_p(st_u64, KECCAK_F_ROUND_COUNT);
    transmute_inverse(st_u64, state);

    pos = 0;
    pos_begin = 0;
}

void Strobe128::absorb(const uint8_t* data, size_t data_len) {
    for (size_t i = 0; i < data_len; ++i) {
        state[pos] ^= data[i];
        pos += 1;
        if (pos == STROBE_R) {
            run_f();
        }
    }
}

void Strobe128::squeeze(uint8_t* data, size_t data_len) {
    for (size_t i = 0; i < data_len; ++i) {
        data[i] = state[pos];
        state[pos] = 0;
        pos += 1;
        if (pos == STROBE_R) {
            run_f();
        }
    }
}

void Strobe128::begin_op(int flags, bool more) {
    if (more) {
        assert(cur_flags == flags && "You tried to continue op but changed flags");
        return;
    }

    assert((flags & FLAG_T) == 0 && "You used the T flag, which this implementation doesn't support");

    int old_begin = pos_begin;
    pos_begin = pos + 1;
    cur_flags = flags;

    uint8_t absorb_data[] = { static_cast<uint8_t>(old_begin), static_cast<uint8_t>(flags) };
    absorb(absorb_data, sizeof(absorb_data));

    bool force_f = (flags & (FLAG_C | FLAG_K)) != 0;

    if (force_f && pos != 0) {
        run_f();
    }
}

void Strobe128::meta_ad(const char* data, bool more) {
    begin_op(FLAG_M | FLAG_A, more);
    absorb(reinterpret_cast<const uint8_t*>(data), strlen(data));
}

void Strobe128::ad(const char* data, bool more) {
    begin_op(FLAG_A, more);
    absorb(reinterpret_cast<const uint8_t*>(data), strlen(data));
}

void Strobe128::prf(uint8_t* data, size_t data_len, bool more) {
    int flags = FLAG_I | FLAG_A | FLAG_C;
    begin_op(flags, more);
    squeeze(data, data_len);
}