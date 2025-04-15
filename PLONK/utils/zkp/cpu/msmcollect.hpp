#pragma once
#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include "collect.h"
#include "ec/jacobian_t.hpp"
#include "ec/xyzz_t.hpp"
#include "../../../../caffe/interface.hpp"
#include "../../mont/cpu/curve_def.h"

using namespace caffe;
#pragma clang diagnostic ignored "-Wmissing-prototypes"
namespace cpu{
    SyncedMemory msm_collect_cpu(SyncedMemory step1res, int64_t npoints);
}