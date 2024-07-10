#include <cstdint>
#include <cstring>
#include "PLONK/src/transcript/strobe.h"
#include "PLONK/src/bls12_381/fr.hpp"
#include "PLONK/src/serialize.cuh"


#define MERLIN_PROTOCOL_LABEL "Merlin v1.0"

class Transcript {
public:
    Strobe128 strobe;
    Transcript(char* label) {
        strobe.new_instance(MERLIN_PROTOCOL_LABEL);
        append_message("dom-sep", reinterpret_cast<uint8_t*>(label));
    }

    void append_message(char* label, uint8_t* message) {
        uint32_t data_len = static_cast<uint32_t>(strlen(label));
        strobe.meta_ad(label, false);
        strobe.meta_ad(reinterpret_cast<char*>(&data_len), true);
        strobe.ad(reinterpret_cast<char*>(message), false);
    }

    void append_pi(char* label, SyncedMemory& item, size_t pos) {
        uint8_t* buf = nullptr;
        serialize(buf, BTreeMap(item,pos));
        append_message(label, buf);
        free(buf);
    }

    void append(char* label, SyncedMemory& item) {
        uint8_t* buf = nullptr;
        serialize(buf, item);
        append_message(label, buf);
        free(buf);
    }

    void challenge_bytes(char* label, char* dest, size_t dest_len) {
        strobe.meta_ad(label, false);
        strobe.meta_ad(reinterpret_cast<char*>(&dest_len), true);
        strobe.prf(reinterpret_cast<uint8_t*>(dest), dest_len, false);
    }

    SyncedMemory& challenge_scalar(char* label) {
        size_t size = fr::MODULUS_BITS / 8;
        uint8_t* buf = static_cast<uint8_t*>(calloc(size, sizeof(uint8_t)));
        challenge_bytes(label, reinterpret_cast<char*>(buf), size);
        SyncedMemory& c_s = deserialize(buf, size);
        free(buf);
        return c_s;
    }
};
