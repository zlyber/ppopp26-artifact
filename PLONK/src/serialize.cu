#include "PLONK/src/serialize.cuh"

bool IsBTreeMap(BTreeMap v) // match pointer  
{  
    return true;  
}  
  
bool IsBTreeMap(...)  // match non-pointer  
{  
    return false;  
} 

// Serialize function
template<typename T>
void serialize(uint8_t* buffer, T& item, int flag) {
    if (sizeof(item) == 40) {
        assert(flag <= 8 && "not enough space");
        SyncedMemory& item_base = to_base(item);
        void* item_ = item_base.mutable_cpu_data();
        memcpy(buffer, item_, item.size());
        buffer[-1] |= flag;
        return;
    }
    else if(sizeof(item) == 8) {
        memcpy(buffer, &item, 8);
        return;
    }
    else if (sizeof(item) == 16) {
        if (IsBTreeMap(item)){
            uint64_t len = 1;
            serialize(buffer, len);
            serialize(buffer + 8, item.pos);
            serialize(buffer + 16, item.item);
            return;
        }
        else{
            if (AffinePointG1::is_zero(item)) {
                flag = SWFlags::infinity().flag;
                serialize(buffer, fq::zero(), flag);
            } else {
                SyncedMemory& a = to_base(item.y);
                SyncedMemory& b = to_base(neg_mod(item.y));
                flag = SWFlags::from_y_sign(gt_zkp(a, b)).flag;
                serialize(buffer, item.x, flag);
            }
            return;
        }
    }
    throw std::runtime_error("unsupported type");
}

// Deserialize function
SyncedMemory& deserialize(uint8_t* x, size_t length) {
    assert(EmptyFlags::BIT_SIZE <= 8 && "empty flags too large");

    uint8_t aligned[64] = {0};
    memcpy(aligned, x, length);
    memset(aligned + length, 0, 64 - length);

    SyncedMemory scalar_in_uint64(sizeof(uint64_t)*fr::Limbs);
    void* scalar_ = scalar_in_uint64.mutable_cpu_data();
    memcpy(scalar_, aligned, sizeof(scalar_in_uint64));
    return to_mont(scalar_in_uint64);
}
