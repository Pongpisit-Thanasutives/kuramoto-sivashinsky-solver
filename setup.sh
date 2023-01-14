### 2023 DEMO ###
TMPDIR="../tmp" pip install -U torch
pip install -U numpy # if it solves conflict
pip install -r requirements.txt

### pub ### 
# pip install -U torch # (now not proper)
# TMPDIR="../tmp" pip install "torch==1.10.1+cu113" -f https://download.pytorch.org/whl/cu113/torch_stable.html # (no need TMPDIR="../tmp" and -f ... in the past)
# pip install -r requirements.txt

# python3 deephpm_KS_chaotic_fixed_coeffs.py
# python3 qho_pinn_fixed_coeffs.py
# python3 reproduced_burgers.py
# python3 ks_selector.py
# python3 qho.py
# python3 qho_pinn_learned_coeffs.py
# python3 qho_pinn_learned_coeffs_20220613.py
# python3 nls_pinn_learned_coeffs_work.py
# python3 deephpm_KS_chaotic_learned_coeffs_noise_old.py
# python3 deephpm_KS_chaotic_fixed_coeffs.py
# python3 deephpm_KS_chaotic_learned_coeffs_cleanall.py
python3 kdv_pinn_2000_pub.py # for cleanall KdV
# python3 nls_pinn_learned_coeffs_new.py
# python3 nls.py
# python3 nls_pinn_learned_coeffs_20220614.py
# python3 ks_selector_100000.py
# python3 deephpm_KS_chaotic_learned_coeffs_noise_new.py
## python3 deephpm_KS_chaotic_learned_coeffs_cleanall_new.py
# python3 deephpm_KS_chaotic_learned_coeffs_more_noise.py
# python3 kdv_pinn_2000_pub_20220517.py # for noisy1 and noisy2 KdV
