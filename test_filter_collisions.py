torque_limits: dict[str, float] = {
    "base_yaw_joint": 1000.0,
    "head_j1": 150.0,
    "head_j2": 150.0,
    "head_j3": 150.0,
    "R_arm_j1": 150.0,
    "R_arm_j2": 150.0,
    "R_arm_j3": 150.0,
    "R_arm_j4": 150.0,
    "R_arm_j5": 150.0,
    "R_arm_j6": 150.0,
    "R_arm_j7": 150.0,
    "R_th_j0": 100.0,
    "R_th_j1": 100.0,
    "R_th_j2": 100.0,
    "R_ff_j1": 100.0,
    "R_ff_j2": 100.0,
    "R_mf_j1": 100.0,
    "R_mf_j2": 100.0,
    "R_rf_j1": 100.0,
    "R_rf_j2": 100.0,
    "R_lf_j1": 100.0,
    "R_lf_j2": 100.0,
}

torque_limits.pop("head_j3", None)
print(torque_limits)