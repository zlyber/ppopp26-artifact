#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include "PLONK/src/transcript/flags.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/fr.hpp"
using namespace caffe;

template<typename T>
void serialize(uint8_t* buffer, T& item, int flag = EmptyFlags::BIT_SIZE);

SyncedMemory& deserialize(uint8_t* x, size_t length);