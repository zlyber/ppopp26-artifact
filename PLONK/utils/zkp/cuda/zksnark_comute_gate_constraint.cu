#include "zksnark.cuh"

namespace cuda{

template <typename T>
__global__ void allmerge_kernel(const int64_t N, 
const T*__restrict__ a_val, const T*__restrict__ b_val, const T*__restrict__ q_m, const T*__restrict__ q_l, 
const T*__restrict__ q_r, const T*__restrict__ c_val, const T*__restrict__ q_o, const T*__restrict__ d_val, 
const T*__restrict__ q_4, const T*__restrict__ q_hl, const T*__restrict__ q_hr, const T*__restrict__ q_h4, 
const T*__restrict__ q_c, const T*__restrict__ q_arith, const T*__restrict__ d_next_eval, const T*__restrict__ range, 
const T*__restrict__ a_next_eval, const T*__restrict__ b_next_eval, const T*__restrict__ logic, const T*__restrict__ fixed_group_add, 
const T*__restrict__ variable_group_add, const T*__restrict__ pi_eval_8n, 
T*__restrict__ gate_contributions, 
const T*__restrict__ four, const T*__restrict__ one, const T*__restrict__ two, const T*__restrict__ three, 
const T*__restrict__ kappa_range, const T*__restrict__ kappa_sq_range, const T*__restrict__ kappa_cu_range, const T*__restrict__ range_challenge, 
const T*__restrict__ P_D, const T*__restrict__ nine, const T*__restrict__ kappa_cu_logic, 
const T*__restrict__ eighteen, const T*__restrict__ eightyone, const T*__restrict__ eightythree, const T*__restrict__ kappa_qu_logic, 
const T*__restrict__ logic_challenge, const T*__restrict__ P_A, const T*__restrict__ kappa_fb, const T*__restrict__ kappa_sq_fb, 
const T*__restrict__ kappa_cu_fb, const T*__restrict__ fb_challenge, const T*__restrict__ kappa_vb, const T*__restrict__ kappa_sq_vb, 
const T*__restrict__ vb_challenge, int64_t SBOX_ALPHA)
{
        int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < N)
{
        gate_contributions[tid] = (((((((c_val[tid]  -  (q_c[tid]  *  (d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))))  *  kappa_fb[0])  +  ((((b_next_eval[tid]  -  ((b_val[tid]  *  (a_val[tid]  *  (c_val[tid]  *  b_next_eval[tid])))  *  P_D[0]))  -  ((b_val[tid]  *  (((q_r[tid]  *  one[0])  *  ((d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))  ^  2))  *  one[0]))  -  (a_val[tid]  *  ((q_l[tid]  *  (d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid])))  *  P_A[0]))))  *  kappa_cu_fb[0])  +  ((((d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))  *  one[0])  *  ((d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))  *  ((d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))  *  one[0])))  +  (((a_next_eval[tid]  +  ((b_val[tid]  *  (a_val[tid]  *  (c_val[tid]  *  a_next_eval[tid])))  *  P_D[0]))  -  ((b_val[tid]  *  (q_l[tid]  *  (d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))))  +  (a_val[tid]  *  (((q_r[tid]  *  one[0])  *  ((d_val[tid]  -  (d_val[tid]  -  d_next_eval[tid]))  ^  2))  *  one[0]))))  *  kappa_sq_fb[0]))))  *  fb_challenge[0])  *  fixed_group_add[tid])  +  ((((((((((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  +  (((a_val[tid]  *  four[0])  -  a_next_eval[tid])  +  ((b_val[tid]  *  four[0])  -  b_next_eval[tid])))  *  three[0])  -  ((c_val[tid]  *  ((((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  +  ((b_val[tid]  *  four[0])  -  b_next_eval[tid]))  *  eightyone[0])  -  ((c_val[tid]  *  (((c_val[tid]  *  four[0])  -  ((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  +  ((b_val[tid]  *  four[0])  -  b_next_eval[tid]))  *  eighteen[0]))  *  eightyone[0]))  +  (((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  ^  2)  +  (((a_val[tid]  *  four[0])  -  a_next_eval[tid])  ^  2))  *  eighteen[0])))  *  eightythree[0]))  *  two[0]))  +  (q_c[tid]  *  (((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  *  nine[0])  -  ((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  +  ((b_val[tid]  *  four[0])  -  b_next_eval[tid]))  *  three[0]))))  *  kappa_qu_logic[0])  +  (((c_val[tid]  -  (((a_val[tid]  *  four[0])  -  a_next_eval[tid])  *  ((b_val[tid]  *  four[0])  -  b_next_eval[tid])))  *  kappa_cu_logic[0])  +  (((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  *  (((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  *  three[0])  *  (((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  *  one[0])  *  ((d_next_eval[tid]  -  (d_val[tid]  *  four[0]))  *  two[0]))))  +  ((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  *  ((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  *  three[0])  *  ((((a_val[tid]  *  four[0])  -  a_next_eval[tid])  *  one[0])  *  (((a_val[tid]  *  four[0])  -  a_next_eval[tid])  *  two[0]))))  +  (((b_val[tid]  *  four[0])  -  b_next_eval[tid])  *  ((((b_val[tid]  *  four[0])  -  b_next_eval[tid])  *  three[0])  *  ((((b_val[tid]  *  four[0])  -  b_next_eval[tid])  *  one[0])  *  (((b_val[tid]  *  four[0])  -  b_next_eval[tid])  *  two[0]))))))))  *  logic_challenge[0])  *  logic[tid])  +  ((((((((a_val[tid]  *  four[0])  -  d_next_eval[tid])  *  ((((a_val[tid]  *  four[0])  -  d_next_eval[tid])  *  three[0])  *  ((((a_val[tid]  *  four[0])  -  d_next_eval[tid])  *  one[0])  *  (((a_val[tid]  *  four[0])  -  d_next_eval[tid])  *  two[0]))))  *  kappa_cu_range[0])  +  ((((a_val[tid]  -  (b_val[tid]  *  four[0]))  *  (((a_val[tid]  -  (b_val[tid]  *  four[0]))  *  three[0])  *  (((a_val[tid]  -  (b_val[tid]  *  four[0]))  *  one[0])  *  ((a_val[tid]  -  (b_val[tid]  *  four[0]))  *  two[0]))))  *  kappa_sq_range[0])  +  (((c_val[tid]  -  (d_val[tid]  *  four[0]))  *  (((c_val[tid]  -  (d_val[tid]  *  four[0]))  *  three[0])  *  (((c_val[tid]  -  (d_val[tid]  *  four[0]))  *  one[0])  *  ((c_val[tid]  -  (d_val[tid]  *  four[0]))  *  two[0]))))  +  (((b_val[tid]  -  (c_val[tid]  *  four[0]))  *  (((b_val[tid]  -  (c_val[tid]  *  four[0]))  *  three[0])  *  (((b_val[tid]  -  (c_val[tid]  *  four[0]))  *  one[0])  *  ((b_val[tid]  -  (c_val[tid]  *  four[0]))  *  two[0]))))  *  kappa_range[0]))))  *  range_challenge[0])  *  range[tid])  +  ((((((d_val[tid]  ^  SBOX_ALPHA)  *  q_h4[tid])  +  (((b_val[tid]  ^  SBOX_ALPHA)  *  q_hr[tid])  +  (((a_val[tid]  ^  SBOX_ALPHA)  *  q_hl[tid])  +  ((d_val[tid]  *  q_4[tid])  +  ((c_val[tid]  *  q_o[tid])  +  ((b_val[tid]  *  q_r[tid])  +  (((a_val[tid]  *  b_val[tid])  *  q_m[tid])  +  (a_val[tid]  *  q_l[tid]))))))))  +  q_c[tid])  *  q_arith[tid])  +  pi_eval_8n[tid]))))  +  (((((((b_val[tid]  *  d_val[tid])  -  ((a_val[tid]  *  c_val[tid])  *  P_A[0]))  -  (b_next_eval[tid]  -  ((b_val[tid]  *  c_val[tid])  *  (d_next_eval[tid]  *  (b_next_eval[tid]  *  P_D[0])))))  *  kappa_sq_vb[0])  +  ((d_next_eval[tid]  -  (a_val[tid]  *  d_val[tid]))  +  (((d_next_eval[tid]  +  (b_val[tid]  *  c_val[tid]))  -  (a_next_eval[tid]  +  ((b_val[tid]  *  c_val[tid])  *  (d_next_eval[tid]  *  (a_next_eval[tid]  *  P_D[0])))))  *  kappa_vb[0])))  *  vb_challenge[0])  *  variable_group_add[tid]));
}
}

SyncedMemory compute_gate_constraint_allmerge_cuda(SyncedMemory a_val, SyncedMemory b_val, SyncedMemory c_val, SyncedMemory d_val, 
SyncedMemory a_next_eval, SyncedMemory b_next_eval, SyncedMemory d_next_eval, 
SyncedMemory q_m, SyncedMemory q_l, SyncedMemory q_r, SyncedMemory q_o, 
SyncedMemory q_4, SyncedMemory q_c, SyncedMemory q_hl, SyncedMemory q_hr, SyncedMemory q_h4, SyncedMemory q_arith, 
SyncedMemory range, SyncedMemory logic, SyncedMemory fixed_group_add, SyncedMemory variable_group_add, SyncedMemory pi_eval_8n, 
SyncedMemory four, SyncedMemory one, SyncedMemory two, SyncedMemory three, 
SyncedMemory kappa_range, SyncedMemory kappa_sq_range, SyncedMemory kappa_cu_range, SyncedMemory range_challenge, 
SyncedMemory P_D, SyncedMemory nine, SyncedMemory kappa_cu_logic, 
SyncedMemory eighteen, SyncedMemory eightyone, SyncedMemory eightythree, SyncedMemory kappa_qu_logic, 
SyncedMemory logic_challenge, SyncedMemory P_A, SyncedMemory kappa_fb, SyncedMemory kappa_sq_fb, 
SyncedMemory kappa_cu_fb, SyncedMemory fb_challenge, SyncedMemory kappa_vb, SyncedMemory kappa_sq_vb, 
SyncedMemory vb_challenge, int64_t SBOX_ALPHA, cudaStream_t stream)
{
        int64_t N = a_val.size()/(fr_LIMBS * sizeof(uint64_t));
        SyncedMemory gate_contributions(a_val.size());

        void* a_val_ = a_val.mutable_gpu_data_async(stream);
        void* b_val_ = b_val.mutable_gpu_data_async(stream);
        void* q_m_ = q_m.mutable_gpu_data_async(stream);
        void* q_l_ = q_l.mutable_gpu_data_async(stream);
        void* q_r_ = q_r.mutable_gpu_data_async(stream);
        void* c_val_ = c_val.mutable_gpu_data_async(stream);
        void* q_o_ = q_o.mutable_gpu_data_async(stream);
        void* d_val_ = d_val.mutable_gpu_data_async(stream);
        void* q_4_ = q_4.mutable_gpu_data_async(stream);
        void* q_hl_ = q_hl.mutable_gpu_data_async(stream);
        void* q_hr_ = q_hr.mutable_gpu_data_async(stream);
        void* q_h4_ = q_h4.mutable_gpu_data_async(stream);
        void* q_c_ = q_c.mutable_gpu_data_async(stream);
        void* q_arith_ = q_arith.mutable_gpu_data_async(stream);
        void* d_next_eval_ = d_next_eval.mutable_gpu_data_async(stream);
        void* range_ = range.mutable_gpu_data_async(stream);
        void* a_next_eval_ = a_next_eval.mutable_gpu_data_async(stream);
        void* b_next_eval_ = b_next_eval.mutable_gpu_data_async(stream);
        void* logic_ = logic.mutable_gpu_data_async(stream);
        void* fixed_group_add_ = fixed_group_add.mutable_gpu_data_async(stream);
        void* variable_group_add_ = variable_group_add.mutable_gpu_data_async(stream);
        void* pi_eval_8n_ = pi_eval_8n.mutable_gpu_data_async(stream);
        void* gate_contributions_ = gate_contributions.mutable_gpu_data_async(stream);
        void* four_ = four.mutable_gpu_data_async(stream);
        void* one_ = one.mutable_gpu_data_async(stream);
        void* two_ = two.mutable_gpu_data_async(stream);
        void* three_ = three.mutable_gpu_data_async(stream);
        void* kappa_range_ = kappa_range.mutable_gpu_data_async(stream);
        void* kappa_sq_range_ = kappa_sq_range.mutable_gpu_data_async(stream);
        void* kappa_cu_range_ = kappa_cu_range.mutable_gpu_data_async(stream);
        void* range_challenge_ = range_challenge.mutable_gpu_data_async(stream);
        void* P_D_ = P_D.mutable_gpu_data_async(stream);
        void* nine_ = nine.mutable_gpu_data_async(stream);
        void* kappa_cu_logic_ = kappa_cu_logic.mutable_gpu_data_async(stream);
        void* eighteen_ = eighteen.mutable_gpu_data_async(stream);
        void* eightyone_ = eightyone.mutable_gpu_data_async(stream);
        void* eightythree_ = eightythree.mutable_gpu_data_async(stream);
        void* kappa_qu_logic_ = kappa_qu_logic.mutable_gpu_data_async(stream);
        void* logic_challenge_ = logic_challenge.mutable_gpu_data_async(stream);
        void* P_A_ = P_A.mutable_gpu_data_async(stream);
        void* kappa_fb_ = kappa_fb.mutable_gpu_data_async(stream);
        void* kappa_sq_fb_ = kappa_sq_fb.mutable_gpu_data_async(stream);
        void* kappa_cu_fb_ = kappa_cu_fb.mutable_gpu_data_async(stream);
        void* fb_challenge_ = fb_challenge.mutable_gpu_data_async(stream);
        void* kappa_vb_ = kappa_vb.mutable_gpu_data_async(stream);
        void* kappa_sq_vb_ = kappa_sq_vb.mutable_gpu_data_async(stream);
        void* vb_challenge_ = vb_challenge.mutable_gpu_data_async(stream);
        int64_t grid = (N + 128 - 1) / 128;
    
        allmerge_kernel<<<grid, 128, 0, stream>>>(N,
        reinterpret_cast<fr*>(a_val_), reinterpret_cast<fr*>(b_val_), reinterpret_cast<fr*>(q_m_), reinterpret_cast<fr*>(q_l_), 
        reinterpret_cast<fr*>(q_r_), reinterpret_cast<fr*>(c_val_), reinterpret_cast<fr*>(q_o_), reinterpret_cast<fr*>(d_val_), 
        reinterpret_cast<fr*>(q_4_), reinterpret_cast<fr*>(q_hl_), reinterpret_cast<fr*>(q_hr_), reinterpret_cast<fr*>(q_h4_), 
        reinterpret_cast<fr*>(q_c_), reinterpret_cast<fr*>(q_arith_), reinterpret_cast<fr*>(d_next_eval_), reinterpret_cast<fr*>(range_), 
        reinterpret_cast<fr*>(a_next_eval_), reinterpret_cast<fr*>(b_next_eval_), reinterpret_cast<fr*>(logic_), reinterpret_cast<fr*>(fixed_group_add_), 
        reinterpret_cast<fr*>(variable_group_add_), reinterpret_cast<fr*>(pi_eval_8n_), reinterpret_cast<fr*>(gate_contributions_), reinterpret_cast<fr*>(four_), reinterpret_cast<fr*>(one_), reinterpret_cast<fr*>(two_), reinterpret_cast<fr*>(three_), 
        reinterpret_cast<fr*>(kappa_range_), reinterpret_cast<fr*>(kappa_sq_range_), reinterpret_cast<fr*>(kappa_cu_range_), reinterpret_cast<fr*>(range_challenge_), 
        reinterpret_cast<fr*>(P_D_), reinterpret_cast<fr*>(nine_), reinterpret_cast<fr*>(kappa_cu_logic_), 
        reinterpret_cast<fr*>(eighteen_), reinterpret_cast<fr*>(eightyone_), reinterpret_cast<fr*>(eightythree_), reinterpret_cast<fr*>(kappa_qu_logic_), 
        reinterpret_cast<fr*>(logic_challenge_), reinterpret_cast<fr*>(P_A_), reinterpret_cast<fr*>(kappa_fb_), reinterpret_cast<fr*>(kappa_sq_fb_), 
        reinterpret_cast<fr*>(kappa_cu_fb_), reinterpret_cast<fr*>(fb_challenge_), reinterpret_cast<fr*>(kappa_vb_), reinterpret_cast<fr*>(kappa_sq_vb_), 
        reinterpret_cast<fr*>(vb_challenge_), SBOX_ALPHA);
       

        return gate_contributions;
}
}//namespace::cuda