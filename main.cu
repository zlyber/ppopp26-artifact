#include <stdint.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "caffe/interface.hpp"
#include "caffe/utils/math_functions.hpp"
#include "PLONK/src/structure.cuh"
#include "PLONK/src/bls12_381/edwards.cuh"
#include <string.h>  // for memcpy

void read_file(const char* filename, void* data){
   // 打开文件
   std::ifstream file(filename, std::ios::binary);
   if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
   }
   file.seekg(0, std::ios::end);
   size_t fileSize = file.tellg();
   file.seekg(0, std::ios::beg);

   file.read(reinterpret_cast<char*>(data), fileSize);
   if (!file) {
    std::cerr << "Error reading file: " << filename << std::endl;
   }
   file.close();
}  



int main(){
   const char* w_l_f = "/home/zhiyuan/w_l-15.bin";
   const char* w_r_f = "/home/zhiyuan/w_r-15.bin";
   const char* w_o_f = "/home/zhiyuan/w_o-15.bin";
   const char* w_4_f = "/home/zhiyuan/w_4-15.bin";

   const char* pp_f = "/home/zhiyuan/params-15.bin";

   uint64_t N = 1<<22;
   uint64_t coset_N = 1<<25;


   SyncedMemory w_l(N*fr::Limbs*sizeof(uint64_t));
   SyncedMemory w_r(N*fr::Limbs*sizeof(uint64_t));
   SyncedMemory w_o(N*fr::Limbs*sizeof(uint64_t));
   SyncedMemory w_4(N*fr::Limbs*sizeof(uint64_t)); 

   SyncedMemory pp(N*fq::Limbs*sizeof(uint64_t)*2);


   void* w_l_ = w_l.mutable_cpu_data();
   void* w_r_ = w_r.mutable_cpu_data();
   void* w_o_ = w_o.mutable_cpu_data();
   void* w_4_ = w_4.mutable_cpu_data(); 

   void* pp_ = pp.mutable_cpu_data();

   
   read_file(w_l_f, w_l_);
   read_file(w_r_f, w_r_);
   read_file(w_o_f, w_o_);
   read_file(w_4_f, w_4_);


   void* w_l__ = w_l.mutable_gpu_data();
   void* w_r__ = w_r.mutable_gpu_data();
   void* w_o__ = w_o.mutable_gpu_data();
   void* w_4__ = w_4.mutable_gpu_data(); 
   std::vector<SyncedMemory> w_polys;
   w_polys.push_back(w_l);
   w_polys.push_back(w_r);
   w_polys.push_back(w_o);
   w_polys.push_back(w_4);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start, 0);
   for (int i=0;i<4;i++){
      SyncedMemory step1_res = cuda::msm_zkp_cuda(pp, w_polys[i], 114);
   }
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop); 
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   std::cout << "Time taken by kernel: " << milliseconds << " ms" << std::endl;
   return 0;
}
