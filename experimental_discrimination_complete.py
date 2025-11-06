"""
Experimental Discrimination: Shell Structure vs Standard Decoherence
=====================================================================
Simulates multiple experimental "runs" with varying environmental conditions
to demonstrate:
1. Shell structure prediction: ν stable, Γ₀ varies
2. Standard decoherence prediction: Both ν and Γ₀ vary together
3. Statistical quantification of their independence

Author: Generated for "Radial Shell Structure of Configuration Space"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(123)

print("=" * 80)
print("EXPERIMENTAL DISCRIMINATION: SHELL STRUCTURE VS STANDARD DECOHERENCE")
print("=" * 80)

# ==========================================
# SIMULATION PARAMETERS
# ==========================================

# Transmon parameters
nu_true_shell = 5e4  # Shell structure prediction: geometric constant (s⁻¹/GHz)
Gamma0_base = 1e4     # Base environmental decoherence (s⁻¹)

# Experimental runs with varying environmental conditions
n_runs = 12  # Number of measurement campaigns
n_transitions = 20  # Transitions measured per run

# Generate realistic transition frequencies
f_01 = 5.0  # GHz (fundamental)
alpha = -0.3  # GHz (anharmonicity)
transition_freqs = np.array([f_01 + n * alpha for n in range(n_transitions)])
transition_freqs = np.maximum(transition_freqs, 0.5)

# ==========================================
# SCENARIO 1: SHELL STRUCTURE PREDICTION
# ==========================================
print("\n### SCENARIO 1: SHELL STRUCTURE (Geometric ν) ###\n")

# Environmental variance increases across runs
# ν stays constant, Γ₀ varies due to:
# - Temperature fluctuations
# - Bath impedance changes
# - Electromagnetic environment drift

Gamma0_variance = np.linspace(0.5, 2.0, n_runs)  # Variance factor
nu_stability = 0.02  # 2% intrinsic noise in ν measurement

# Storage for fit results
nu_fits_shell = []
Gamma0_fits_shell = []
nu_errors_shell = []
Gamma0_errors_shell = []

print("Simulating experimental runs with varying Γ₀...")
print(f"{'Run':<6} {'Γ₀ Factor':<12} {'Fitted ν':<15} {'Fitted Γ₀':<15}")
print("-" * 60)

for run in range(n_runs):
    # ν remains constant (with small measurement noise)
    nu_this_run = nu_true_shell * (1 + np.random.normal(0, nu_stability))
    
    # Γ₀ varies significantly
    Gamma0_this_run = Gamma0_base * Gamma0_variance[run] * (1 + np.random.normal(0, 0.2))
    
    # True decoherence for this run
    Gamma_true = nu_this_run * transition_freqs + Gamma0_this_run
    
    # Add measurement noise (5%)
    noise = np.random.normal(0, 0.05, n_transitions)
    Gamma_measured = Gamma_true * (1 + noise)
    
    # Fit linear model
    def linear_model(f, nu, Gamma0):
        return nu * f + Gamma0
    
    params, cov = curve_fit(linear_model, transition_freqs, Gamma_measured,
                           p0=[4e4, 1e4])
    nu_fit, Gamma0_fit = params
    perr = np.sqrt(np.diag(cov))
    
    nu_fits_shell.append(nu_fit)
    Gamma0_fits_shell.append(Gamma0_fit)
    nu_errors_shell.append(perr[0])
    Gamma0_errors_shell.append(perr[1])
    
    print(f"{run+1:<6} {Gamma0_variance[run]:<12.2f} {nu_fit/1e3:<15.1f} {Gamma0_fit/1e3:<15.1f}")

nu_fits_shell = np.array(nu_fits_shell)
Gamma0_fits_shell = np.array(Gamma0_fits_shell)
nu_errors_shell = np.array(nu_errors_shell)
Gamma0_errors_shell = np.array(Gamma0_errors_shell)

# Statistical analysis
nu_mean_shell = np.mean(nu_fits_shell)
nu_std_shell = np.std(nu_fits_shell)
Gamma0_mean_shell = np.mean(Gamma0_fits_shell)
Gamma0_std_shell = np.std(Gamma0_fits_shell)

print(f"\nStatistical Summary (Shell Structure):")
print(f"  ν: {nu_mean_shell/1e3:.1f} ± {nu_std_shell/1e3:.1f} (10³ s⁻¹/GHz)")
print(f"  Relative stability: {nu_std_shell/nu_mean_shell*100:.1f}%")
print(f"  Γ₀: {Gamma0_mean_shell/1e3:.1f} ± {Gamma0_std_shell/1e3:.1f} (10³ s⁻¹)")
print(f"  Relative variability: {Gamma0_std_shell/Gamma0_mean_shell*100:.0f}%")

# Correlation test
correlation_shell, p_value_shell = pearsonr(nu_fits_shell, Gamma0_fits_shell)
print(f"\nCorrelation test:")
print(f"  Pearson r(ν, Γ₀) = {correlation_shell:.3f} (p = {p_value_shell:.3f})")
print(f"  Interpretation: {'INDEPENDENT' if abs(correlation_shell) < 0.3 else 'CORRELATED'}")

# ==========================================
# SCENARIO 2: STANDARD DECOHERENCE
# ==========================================
print("\n### SCENARIO 2: STANDARD OHMIC BATH DECOHERENCE ###\n")

# In standard model, BOTH ν and Γ₀ depend on bath
# ν ∝ η (spectral density strength)
# Γ₀ ∝ low-frequency bath coupling

nu_base_standard = 5e4
correlation_factor = 0.7  # Strong correlation between ν and Γ₀

nu_fits_standard = []
Gamma0_fits_standard = []
nu_errors_standard = []
Gamma0_errors_standard = []

print("Simulating standard decoherence with bath variations...")
print(f"{'Run':<6} {'Bath η':<12} {'Fitted ν':<15} {'Fitted Γ₀':<15}")
print("-" * 60)

for run in range(n_runs):
    # Bath strength varies
    bath_strength = 0.8 + 0.4 * run / n_runs
    
    # Both ν and Γ₀ scale with bath (correlated)
    nu_this_run = nu_base_standard * bath_strength * (1 + np.random.normal(0, 0.05))
    Gamma0_this_run = Gamma0_base * bath_strength * (1 + np.random.normal(0, 0.15))
    
    # True decoherence
    Gamma_true = nu_this_run * transition_freqs + Gamma0_this_run
    
    # Measurement noise
    noise = np.random.normal(0, 0.05, n_transitions)
    Gamma_measured = Gamma_true * (1 + noise)
    
    # Fit
    params, cov = curve_fit(linear_model, transition_freqs, Gamma_measured,
                           p0=[4e4, 1e4])
    nu_fit, Gamma0_fit = params
    perr = np.sqrt(np.diag(cov))
    
    nu_fits_standard.append(nu_fit)
    Gamma0_fits_standard.append(Gamma0_fit)
    nu_errors_standard.append(perr[0])
    Gamma0_errors_standard.append(perr[1])
    
    print(f"{run+1:<6} {bath_strength:<12.2f} {nu_fit/1e3:<15.1f} {Gamma0_fit/1e3:<15.1f}")

nu_fits_standard = np.array(nu_fits_standard)
Gamma0_fits_standard = np.array(Gamma0_fits_standard)
nu_errors_standard = np.array(nu_errors_standard)
Gamma0_errors_standard = np.array(Gamma0_errors_standard)

# Statistical analysis
nu_mean_standard = np.mean(nu_fits_standard)
nu_std_standard = np.std(nu_fits_standard)
Gamma0_mean_standard = np.mean(Gamma0_fits_standard)
Gamma0_std_standard = np.std(Gamma0_fits_standard)

print(f"\nStatistical Summary (Standard Decoherence):")
print(f"  ν: {nu_mean_standard/1e3:.1f} ± {nu_std_standard/1e3:.1f} (10³ s⁻¹/GHz)")
print(f"  Relative variability: {nu_std_standard/nu_mean_standard*100:.1f}%")
print(f"  Γ₀: {Gamma0_mean_standard/1e3:.1f} ± {Gamma0_std_standard/1e3:.1f} (10³ s⁻¹)")
print(f"  Relative variability: {Gamma0_std_standard/Gamma0_mean_standard*100:.0f}%")

# Correlation test
correlation_standard, p_value_standard = pearsonr(nu_fits_standard, Gamma0_fits_standard)
print(f"\nCorrelation test:")
print(f"  Pearson r(ν, Γ₀) = {correlation_standard:.3f} (p = {p_value_standard:.3f})")
print(f"  Interpretation: {'INDEPENDENT' if abs(correlation_standard) < 0.3 else 'CORRELATED'}")

# ==========================================
# VISUALIZATION
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ===== Row 1: Shell Structure =====

# Panel A: ν vs run number
axes[0, 0].errorbar(range(1, n_runs+1), nu_fits_shell/1e3, yerr=nu_errors_shell/1e3,
                    fmt='o-', color='blue', capsize=3, markersize=6, label='Measured ν')
axes[0, 0].axhline(nu_true_shell/1e3, color='red', linestyle='--', linewidth=2,
                   label=f'True ν = {nu_true_shell/1e3:.0f}')
axes[0, 0].fill_between(range(1, n_runs+1),
                        (nu_mean_shell - nu_std_shell)/1e3,
                        (nu_mean_shell + nu_std_shell)/1e3,
                        alpha=0.2, color='blue')
axes[0, 0].set_xlabel('Experimental Run', fontsize=11)
axes[0, 0].set_ylabel('ν (10³ s⁻¹/GHz)', fontsize=11)
axes[0, 0].set_title('Shell Structure: ν Stability\n(Geometric Coupling)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel B: Γ₀ vs run number
axes[0, 1].errorbar(range(1, n_runs+1), Gamma0_fits_shell/1e3, yerr=Gamma0_errors_shell/1e3,
                    fmt='s-', color='green', capsize=3, markersize=6, label='Measured Γ₀')
axes[0, 1].axhline(Gamma0_base/1e3, color='red', linestyle='--', linewidth=2,
                   label=f'Base Γ₀ = {Gamma0_base/1e3:.0f}')
axes[0, 1].set_xlabel('Experimental Run', fontsize=11)
axes[0, 1].set_ylabel('Γ₀ (10³ s⁻¹)', fontsize=11)
axes[0, 1].set_title('Shell Structure: Γ₀ Variability\n(Environmental Noise)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel C: Correlation plot
axes[0, 2].scatter(Gamma0_fits_shell/1e3, nu_fits_shell/1e3, c='purple', s=80,
                   alpha=0.6, edgecolors='black', linewidths=1)
# Fit line
z = np.polyfit(Gamma0_fits_shell, nu_fits_shell, 1)
p = np.poly1d(z)
x_line = np.linspace(Gamma0_fits_shell.min(), Gamma0_fits_shell.max(), 100)
axes[0, 2].plot(x_line/1e3, p(x_line)/1e3, 'r--', linewidth=2,
                label=f'r = {correlation_shell:.2f}')
axes[0, 2].set_xlabel('Γ₀ (10³ s⁻¹)', fontsize=11)
axes[0, 2].set_ylabel('ν (10³ s⁻¹/GHz)', fontsize=11)
axes[0, 2].set_title('Shell Structure: ν vs Γ₀\n(No Correlation)', fontsize=12, fontweight='bold')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(True, alpha=0.3)

# ===== Row 2: Standard Decoherence =====

# Panel D: ν vs run number
axes[1, 0].errorbar(range(1, n_runs+1), nu_fits_standard/1e3, yerr=nu_errors_standard/1e3,
                    fmt='o-', color='orange', capsize=3, markersize=6, label='Measured ν')
axes[1, 0].set_xlabel('Experimental Run', fontsize=11)
axes[1, 0].set_ylabel('ν (10³ s⁻¹/GHz)', fontsize=11)
axes[1, 0].set_title('Standard Decoherence: ν Variability\n(Bath-Dependent)', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Panel E: Γ₀ vs run number  
axes[1, 1].errorbar(range(1, n_runs+1), Gamma0_fits_standard/1e3, yerr=Gamma0_errors_standard/1e3,
                    fmt='s-', color='brown', capsize=3, markersize=6, label='Measured Γ₀')
axes[1, 1].set_xlabel('Experimental Run', fontsize=11)
axes[1, 1].set_ylabel('Γ₀ (10³ s⁻¹)', fontsize=11)
axes[1, 1].set_title('Standard Decoherence: Γ₀ Variability\n(Bath-Dependent)', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# Panel F: Correlation plot
axes[1, 2].scatter(Gamma0_fits_standard/1e3, nu_fits_standard/1e3, c='red', s=80,
                   alpha=0.6, edgecolors='black', linewidths=1)
# Fit line
z = np.polyfit(Gamma0_fits_standard, nu_fits_standard, 1)
p = np.poly1d(z)
x_line = np.linspace(Gamma0_fits_standard.min(), Gamma0_fits_standard.max(), 100)
axes[1, 2].plot(x_line/1e3, p(x_line)/1e3, 'b--', linewidth=2,
                label=f'r = {correlation_standard:.2f}')
axes[1, 2].set_xlabel('Γ₀ (10³ s⁻¹)', fontsize=11)
axes[1, 2].set_ylabel('ν (10³ s⁻¹/GHz)', fontsize=11)
axes[1, 2].set_title('Standard Decoherence: ν vs Γ₀\n(Strong Correlation)', fontsize=12, fontweight='bold')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/experimental_discrimination.png', dpi=300, bbox_inches='tight')
print("\n[Figure saved: experimental_discrimination.png]")
plt.show()

# ==========================================
# STATISTICAL SUMMARY TABLE
# ==========================================

print("\n" + "=" * 80)
print("STATISTICAL COMPARISON")
print("=" * 80)

print(f"\n{'Metric':<35} {'Shell Structure':<25} {'Standard Decoherence':<25}")
print("-" * 85)
print(f"{'ν relative stability':<35} {nu_std_shell/nu_mean_shell*100:>8.1f}%{'':<15} {nu_std_standard/nu_mean_standard*100:>8.1f}%")
print(f"{'Γ₀ relative variability':<35} {Gamma0_std_shell/Gamma0_mean_shell*100:>8.0f}%{'':<15} {Gamma0_std_standard/Gamma0_mean_standard*100:>8.0f}%")
print(f"{'Correlation r(ν, Γ₀)':<35} {correlation_shell:>8.3f}{'':<15} {correlation_standard:>8.3f}")
print(f"{'Variance ratio σ(Γ₀)/σ(ν)':<35} {Gamma0_std_shell/nu_std_shell:>8.1f}{'':<15} {Gamma0_std_standard/nu_std_standard:>8.1f}")

print("\n" + "-" * 85)
print("INTERPRETATION:")
print(f"  Shell Structure: ν stable ({nu_std_shell/nu_mean_shell*100:.1f}%) despite Γ₀ varying ({Gamma0_std_shell/Gamma0_mean_shell*100:.0f}%)")
print(f"  → Consistent with geometric coupling (intrinsic to system)")
print(f"\n  Standard Model: Both ν and Γ₀ vary together (r = {correlation_standard:.2f})")
print(f"  → Consistent with bath-dependent decoherence (extrinsic)")

# ==========================================
# EXPERIMENTAL PROTOCOL
# ==========================================

print("\n" + "=" * 80)
print("RECOMMENDED EXPERIMENTAL PROTOCOL")
print("=" * 80)

print("""
Day 1-3: Baseline measurements
  - Measure all 20 transitions at constant bath conditions
  - Repeat 3 times to establish ν₀ and statistical uncertainty
  - Expected precision: δν/ν ≈ 1-2%

Week 2: Temperature variations
  - Vary substrate temperature: 10-50 mK
  - Measure full transition spectrum at each temperature
  - Shell prediction: ν constant, Γ₀ changes
  - Standard prediction: Both vary (thermal occupation changes)

Week 3: Bath impedance engineering
  - Modify readout circuit impedance
  - Tune engineered dissipation via external resistor
  - Shell prediction: ν constant, Γ₀ scales with impedance
  - Standard prediction: ν ∝ η (spectral density), Γ₀ ∝ η

Week 4: Electromagnetic environment
  - Introduce controlled EM noise (filtered sources)
  - Vary pulse powers in control lines
  - Shell prediction: ν unchanged, Γ₀ increases with noise
  - Standard prediction: Both increase

Analysis:
  - Plot ν vs environmental parameter for all conditions
  - Calculate correlation r(ν, Γ₀) across all runs
  - |r| < 0.3 → Shell structure confirmed
  - |r| > 0.6 → Standard decoherence
""")

print("=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
print("\nKey Result: The independence of ν from Γ₀ variance provides a")
print("discriminating experimental signature for geometric shell coupling.")
