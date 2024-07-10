

class FBSMValues {
public:
    // 下一位置的左边电线值
    fr::Fr a_next_val;
    // 下一位置的右边电线值
    fr::Fr b_next_val;
    // 下一位置的第四根电线值
    fr::Fr d_next_val;
    // 左边选择器值
    fr::Fr q_l_val;
    // 右边选择器值
    fr::Fr q_r_val;
    // 常量选择器值
    fr::Fr q_c_val;

    FBSMValues(fr::Fr a_next_val, fr::Fr b_next_val, fr::Fr d_next_val, fr::Fr q_l_val, fr::Fr q_r_val, fr::Fr q_c_val) :
        a_next_val(a_next_val), b_next_val(b_next_val), d_next_val(d_next_val), q_l_val(q_l_val), q_r_val(q_r_val), q_c_val(q_c_val) {}

    static FBSMValues from_evaluations(const std::map<std::string, fr::Fr>& custom_evals) {
        fr::Fr a_next_val = custom_evals.at("a_next_eval");
        fr::Fr b_next_val = custom_evals.at("b_next_eval");
        fr::Fr d_next_val = custom_evals.at("d_next_eval");
        fr::Fr q_l_val = custom_evals.at("q_l_eval");
        fr::Fr q_r_val = custom_evals.at("q_r_eval");
        fr::Fr q_c_val = custom_evals.at("q_c_eval");

        return FBSMValues(a_next_val, b_next_val, d_next_val, q_l_val, q_r_val, q_c_val);
    }
};