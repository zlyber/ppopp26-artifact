#include "transcript.cuh"

std::vector<uint8_t> encode_usize_as_u32(size_t x) {
    assert(x <= static_cast<size_t>(UINT32_MAX));

    std::vector<uint8_t> buf(4);

    uint32_t value = static_cast<uint32_t>(x);
    buf[0] = static_cast<uint8_t>(value & 0xff);
    buf[1] = static_cast<uint8_t>((value >> 8) & 0xff);
    buf[2] = static_cast<uint8_t>((value >> 16) & 0xff);
    buf[3] = static_cast<uint8_t>((value >> 24) & 0xff);

    return buf;
}

// Deserialize function
SyncedMemory deserialize(std::vector<uint8_t> x, size_t length) {
    assert(EmptyFlags::BIT_SIZE <= 8 && "empty flags too large");

    uint8_t aligned[fr::Limbs*8] = {0};
    memcpy(aligned, x.data(), length);
    memset(aligned + length, 0, fr::Limbs*8 - length);

    SyncedMemory scalar_in_uint64(sizeof(uint64_t)*fr::Limbs);
    void* scalar_ = scalar_in_uint64.mutable_cpu_data();
    for(int i = 0; i < fr::Limbs; i++){
        memcpy(scalar_ + sizeof(uint64_t)*i, aligned + 8*i, 8);
    }
    return to_mont(scalar_in_uint64);
}

Transcript::Transcript(std::string label){
    Strobe128 strobe_ = Strobe128::new_instance(MERLIN_PROTOCOL_LABEL);
    strobe = strobe_;
    append_message("dom-sep", label);
}

void Transcript::append_message(std::string label, std::string message) {
    std::vector<uint8_t> data_len = encode_usize_as_u32(message.size());
    std::vector<uint8_t> data = str_to_u8(label);
    strobe.meta_ad(data, false);
    strobe.meta_ad(data_len, true);
    std::vector<uint8_t> data2 = str_to_u8(message);
    strobe.ad(data2, false);
}

void Transcript::append_message_chunk(std::string message) {
    std::vector<uint8_t> data2 = str_to_u8(message);
    strobe.ad(data2, false);
}

void Transcript::append_pi(std::string label, SyncedMemory item, size_t pos) {
    std::vector<uint8_t> buf(48);
    SyncedMemory item_field = to_mont(item);
    serialize(buf, BTreeMap::new_instance(item_field, pos), EmptyFlags(0));
    append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
}

void Transcript::append(char* label, SyncedMemory item) {
    std::vector<uint8_t> buf(item.size());
    serialize(buf, item, EmptyFlags(0));
    append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
}

void Transcript::append_chunk(const char* label, AffinePointG1 item, int idx) {
    std::vector<uint8_t> buf(48);
    serialize(buf, item, EmptyFlags(0));
    if(idx == 0)
        append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
    else
        append_message_chunk(std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
}

void Transcript::append(char* label, AffinePointG1 item) {
    std::vector<uint8_t> buf(48);
    serialize(buf, item, EmptyFlags(0));
    append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
}

void Transcript::challenge_bytes(std::string label, std::vector<uint8_t>& dest) {
    std::vector<uint8_t> data_len = encode_usize_as_u32(dest.size());
    std::vector<uint8_t> data = str_to_u8(label);
    strobe.meta_ad(data, false);
    strobe.meta_ad(data_len, true);
    strobe.prf(dest, false);
}

SyncedMemory Transcript::challenge_scalar(std::string label) {
    size_t size = fr::MODULUS_BITS / 8;
    std::vector<uint8_t> buf(size, 0);
    challenge_bytes(label, buf);
    SyncedMemory c_s = deserialize(buf, size);
    return c_s;
}

