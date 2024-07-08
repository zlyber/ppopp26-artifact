#pragma once
#include <iostream>
#include <vector>
#include "caffe/syncedmem.hpp"
#include "bls12_381/fq.hpp"
#include "PLONK/utils/function.cuh"

constexpr int COEFF_A = 0;

class AffinePointG1{
    public:
        SyncedMemory& x;
        SyncedMemory& y;
        AffinePointG1(SyncedMemory& a, SyncedMemory& b);
        static bool is_zero(AffinePointG1 self);
};

class ProjectivePointG1{
    public:
        SyncedMemory& x;
        SyncedMemory& y;
        SyncedMemory& z;
        ProjectivePointG1(SyncedMemory& a, SyncedMemory& b, SyncedMemory& c);
        static bool is_zero(ProjectivePointG1 self);
};

bool is_zero_ProjectivePointG1(SyncedMemory& self);

AffinePointG1 to_affine(ProjectivePointG1 G1);

ProjectivePointG1 add_assign(ProjectivePointG1 self, ProjectivePointG1 other);

ProjectivePointG1 double_ProjectivePointG1(ProjectivePointG1 self);

ProjectivePointG1 add_assign_mixed(ProjectivePointG1 self, ProjectivePointG1 other);

typedef struct {
    SyncedMemory& item;
    uint64_t pos;
} BTreeMap;
