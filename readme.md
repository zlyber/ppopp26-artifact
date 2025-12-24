**Table of Contents**

- [TL;DR](#tldr)
- [Remark](#remark)
- [Environment Requirements](#environment-requirements)
- [How to Run](#how-to-run)
  - [Evaluate NTT](#evaluate-ntt)
  - [Evaluate LDE](#evaluate-ntt)
  - [Evaluate Missing Operators](#evaluate-missing-operators)
 

### TL;DR
This artifact provides the minimal reproducible examples for *Pipelonk*’s segmentable operator library (sec.3 in the paper), including grand product, polynomial evaluation, polynomial division, number theoretic transforms (NTT), and low degree extension (LDE).
### Remark
- We have made every effort to ensure that the artifact's code correctly implements the logic and computational workload described in the paper. Nonetheless, it is provided as-is, without any guarantees, and should be independently validated before use in real systems.
- The **cpu baseline** results for the missing operators reported in the paper were obtained using the [*arkworks*’ Rust implementation](https://github.com/ZK-Garage/plonk). This artifact re-implements the same logic but uses the high-performance [_blst_](https://github.com/supranational/blst) library, written in assembly and C, resulting in baseline performance that is approximately 25%–55% faster than the data reported in the paper.
### Environment Requirements
This artifact primarily targets x86_64 Linux systems equipped with NVIDIA Volta-class or newer GPUs. We have tested the code on NVIDIA H100 and RTX 4090 using CUDA 12.8 and Ubuntu 24.04. The required environment is listed below:
- CMake ≥ 3.18
- CUDA ≥ 12.0
- C++17 compiler (GCC ≥ 9 recommended)
- Linux OS(UBUNTU ≥20.04 recommended)
### How to Run
Each test group includes both the *Pipelonk*-designed operator (_ours_) and the baseline implementation (_naive_), and each configuration is executed three times.
#### Evaluate NTT
```
1. cd to the artifact directory
2. cd NTT
3. mkdir build & cd build
4. cmake ..
5. make -j
6. ./NTT
```
#### Evaluate LDE
```
1. cd to the artifact directory
2. cd LDE
3. mkdir build & cd build
4. cmake ..
5. make -j
6. ./LDE
```
#### Evaluate Missing Operators
This test includes the grand product, polynomial evaluation, and polynomial division.
```
1. cd to the artifact directory
2. cd miss_ops
3. mkdir build & cd build
4. cmake ..
5. make -j
6. ./miss_ops
7. to test with input scale 2^{24}, change ```lg_N``` in line 90 to     24 and ```file_path``` in line 96 to ""../../input-24.bin""
```
