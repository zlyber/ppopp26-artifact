#pragma once
#include "strobe.h"
#include "../serialize.cuh"

#define MERLIN_PROTOCOL_LABEL "Merlin v1.0"

std::vector<uint8_t> encode_usize_as_u32(size_t x);

SyncedMemory deserialize(std::vector<uint8_t> x, size_t length);

class Transcript {
public:
    Strobe128 strobe;
    Transcript(std::string label);
    
    void append_message(std::string label, std::string message);

    void append_message_chunk(std::string message);

    void append_pi(std::string label, SyncedMemory item, size_t pos);
    
    void append(char* label, SyncedMemory item);

    void append_chunk(const char* label, AffinePointG1 item, int idx);

    void append(char* label, AffinePointG1 item);

    void challenge_bytes(std::string label, std::vector<uint8_t>& dest);

    SyncedMemory challenge_scalar(std::string label);
};

