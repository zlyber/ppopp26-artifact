#pragma once
#include <cstdint>

class SWFlags {
public:
    static const int BIT_SIZE = 2;
    enum flags { Infinity, PositiveY, NegativeY };

    int flag;

    SWFlags(int flag) : flag(flag) {}

    static SWFlags infinity() {
        return SWFlags(Infinity);
    }

    static SWFlags from_y_sign(bool is_positive) {
        if (is_positive) {
            return SWFlags(PositiveY);
        } else {
            return SWFlags(NegativeY);
        }
    }

    uint8_t u8_bitmask() const {
        uint8_t mask = 0;
        if (flag == Infinity) {
            mask |= 1 << 6;
        } else if (flag == PositiveY) {
            mask |= 1 << 7;
        }
        return mask;
    }
};

class EmptyFlags {
public:
    int flag;
    static const int BIT_SIZE = 0;
    EmptyFlags(int flags): flag(flags) {}
    uint8_t u8_bitmask() {
        return 0;
    }
};
