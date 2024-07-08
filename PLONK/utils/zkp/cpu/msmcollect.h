#pragma once
#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include "PLONK/utils/zkp/cpu/collect.h"
#include "PLONK/utils/zkp/cpu/ec/jacobian_t.hpp"
#include "PLONK/utils/zkp/cpu/ec/xyzz_t.hpp"
#include "caffe/syncedmem.hpp"
#include "PLONK/utils/mont/cpu/curve_def.h"

using namespace caffe;
#pragma clang diagnostic ignored "-Wmissing-prototypes"
namespace cpu{
    template<typename T>
    static void pippenger_collect(T* self, T* step1res, size_t npoints, int64_t numel);

    SyncedMemory& msm_collect_cpu(SyncedMemory& step1res, int64_t npoints);
}