# Technical Models and Mathematical Foundations

## Overview

This document provides detailed mathematical descriptions of the PUF models, environmental stressor models, and attack simulation methods implemented in the PPET framework. All models are based on published research and validated against experimental data from the literature.

## PUF Models

### 1. Arbiter PUF Model

**Physical Basis**: The Arbiter PUF is based on race conditions between two identical paths in a silicon switch chain. Manufacturing variations cause slight differences in propagation delays, creating a unique response pattern.

**Mathematical Model**:

The delay difference for an n-stage arbiter PUF is given by:

```
Δt = Σ(i=1 to n) c_i × (Δd_i,top - Δd_i,bottom)
```

Where:
- `c_i` is the challenge bit for stage i (0 or 1)
- `Δd_i,top` is the delay deviation for the top path of stage i
- `Δd_i,bottom` is the delay deviation for the bottom path of stage i

**Hierarchical Variation Model**:

The delay deviations follow a hierarchical model:

```
Δd_i,path = G + S_i + L_i,path + E_i,path(t)
```

Where:
- `G` ~ N(0, σ_g²): Global process variation (chip-wide)
- `S_i` ~ N(0, σ_s²): Systematic variation (position-dependent)
- `L_i,path` ~ N(0, σ_l²): Local random variation (stage and path specific)
- `E_i,path(t)`: Environmental effects (temperature, voltage, aging)

**Environmental Effects**:

Temperature effect:
```
E_temp(T) = α_temp × (T - T_nom)
```

Voltage effect:
```
E_voltage(V) = α_voltage × ((V - V_nom) / V_nom)²
```

Aging effect:
```
E_aging(t) = α_aging × (1 - exp(-t/τ))
```

**Parameters**:
- σ_g = 0.4 (40% global variation)
- σ_s = 0.2 (20% systematic variation)  
- σ_l = 0.3 (30% local variation)
- α_temp = 0.00005 per °C (validated implementation)
- α_voltage = 0.05 per V² (validated implementation)
- α_aging = 0.1 (10% maximum aging)
- τ = 8760 hours (1 year time constant, validated implementation)

**References**:
- Gassend et al. "Silicon Physical Random Functions" (CCS 2002)
- Majzoobi et al. "Testing Techniques for Hardware Security" (ITC 2008)

### 2. SRAM PUF Model

**Physical Basis**: SRAM PUFs exploit the slight imbalance between cross-coupled inverters in SRAM cells, causing cells to prefer one state over another during power-up.

**Mathematical Model**:

The probability that an SRAM cell i powers up to state '1' is:

```
P_i = Φ((V_th,1 - V_th,2) / √(2σ_th²))
```

Where:
- `Φ` is the standard normal cumulative distribution function
- `V_th,1`, `V_th,2` are the threshold voltages of the two transistors
- `σ_th` is the standard deviation of threshold voltage variation

**Threshold Voltage Variation**:

```
V_th,i = V_th,nom + ΔV_th,global + ΔV_th,local,i + ΔV_th,env(T,V)
```

Where:
- `ΔV_th,global` ~ N(0, σ_global²): Global process variation
- `ΔV_th,local,i` ~ N(0, σ_local²): Local random variation
- `ΔV_th,env(T,V)`: Environmental effects

**Environmental Effects**:

Temperature dependence:
```
ΔV_th,temp(T) = α_temp × (T - T_nom)
```

Voltage dependence:
```
ΔV_th,voltage(V) = α_voltage × ln(V/V_nom)
```

**Parameters**:
- σ_global = 0.05 V (global variation)
- σ_local = 0.02 V (local variation)
- α_temp = -0.0005 V/°C
- α_voltage = 0.1 V per ln(V)

**References**:
- Holcomb et al. "Initial SRAM State as a Fingerprint" (HOST 2009)
- Koeberl et al. "The Reliability of SRAM-based PUFs" (ISQED 2013)

### 3. Ring Oscillator PUF Model

**Physical Basis**: Ring oscillator PUFs use the frequency variations of ring oscillators due to manufacturing process variations.

**Mathematical Model**:

The frequency of ring oscillator i is:

```
f_i = 1 / (2 × N × t_d,i)
```

Where:
- `N` is the number of stages in the ring
- `t_d,i` is the average delay per stage

**Delay Variation Model**:

```
t_d,i = t_d,nom × (1 + δ_global + δ_local,i + δ_env(T,V,t))
```

Where:
- `δ_global` ~ N(0, σ_global²): Global process variation
- `δ_local,i` ~ N(0, σ_local²): Local variation for oscillator i
- `δ_env(T,V,t)`: Environmental effects

**Spatial Correlation**:

Ring oscillators exhibit spatial correlation based on their physical proximity:

```
Corr(δ_i, δ_j) = exp(-d_ij / λ)
```

Where:
- `d_ij` is the Euclidean distance between oscillators i and j
- `λ` is the correlation length

**Parameters**:
- σ_global = 0.03 (3% global variation)
- σ_local = 0.05 (5% local variation)
- λ = 100 μm (correlation length)

**References**:
- Suh & Devadas "Physical Unclonable Functions for Device Authentication" (DAC 2007)
- Maiti et al. "Improved Ring Oscillator PUF" (HOST 2011)

## Environmental Stressor Models

### Temperature Effects

**Physical Basis**: Temperature affects semiconductor device parameters through thermal activation and mobility degradation.

**Threshold Voltage Temperature Dependence**:
```
V_th(T) = V_th(T_nom) + α_temp × (T - T_nom)
```

**Mobility Temperature Dependence**:
```
μ(T) = μ_0 × (T / T_nom)^(-α_mobility)
```

**Parameters**:
- α_temp = -0.0005 V/°C (for PMOS)
- α_temp = 0.0003 V/°C (for NMOS)
- α_mobility = 1.5

### Voltage Effects

**Physical Basis**: Supply voltage variations affect drive current and propagation delay through the square-law relationship.

**Delay-Voltage Relationship**:
```
t_d(V) = t_d(V_nom) × (V_nom / V)^α
```

Where α ≈ 1.3 for short-channel devices.

### Aging Effects

**Physical Basis**: Device aging primarily due to Negative Bias Temperature Instability (NBTI) and Hot Carrier Injection (HCI).

**Implemented Aging Model**:
The PPET framework implements a comprehensive aging model combining exponential decay with temperature acceleration:

```
aging_factor = 1.0 + base_aging × temp_acceleration
```

Where:
```
base_aging = α_aging × (1 - exp(-t/τ))
temp_acceleration = cumulative_stress / (time × sampling_points)
```

**Temperature-Dependent Acceleration**:
Using Arrhenius model for temperature dependency:
```
acceleration_factor = exp((Ea/k) × (1/T_ref - 1/T_stress))
```

Where:
- Ea = 0.6 eV (activation energy for silicon device aging)
- k = 8.617333e-5 eV/K (Boltzmann constant)
- T_ref = 323.15 K (50°C reference temperature)
- T_stress = actual temperature in Kelvin

**Implementation Parameters**:
- α_aging = 0.1 (maximum aging factor)
- τ = 8760 hours (1 year time constant)
- base_aging_rate = calibrated per cumulative stress

**NBTI Model** (theoretical reference):
```
ΔV_th(t) = A × (t / t_0)^n
```

Where:
- A is the aging coefficient
- t_0 is the reference time (1 year)
- n ≈ 0.16 (time exponent)

**HCI Model** (theoretical reference):
```
ΔV_th(t) = B × (I_d / W)^m × t^n
```

Where:
- B is the HCI coefficient
- I_d is the drain current
- W is the device width
- m ≈ 2, n ≈ 0.5

## Attack Models

### Machine Learning Attacks

**Linear Model**:
For an n-stage Arbiter PUF, the linear model assumes:

```
y = sign(w^T × φ(c))
```

Where:
- `w` is the weight vector to be learned
- `φ(c)` is the feature vector derived from challenge c
- `y` is the response bit

**Feature Engineering**:
The feature vector includes:
- Linear terms: φ_i = 2c_i - 1
- Non-linear terms: φ_ij = φ_i × φ_j (pairwise products)

**Attack Success Rate**:
The theoretical success rate for a linear attack is:

```
P_success = 1 - Q(√(SNR / 2))
```

Where:
- Q is the Q-function
- SNR is the signal-to-noise ratio of the linear model

### Side-Channel Attacks

**Power Analysis Model**:
The power consumption during PUF evaluation is modeled as:

```
P(t) = P_static + α × HD(s(t)) + n(t)
```

Where:
- P_static is the static power consumption
- α is the switching activity coefficient
- HD(s(t)) is the Hamming distance of the internal state
- n(t) is measurement noise

**Timing Analysis Model**:
The timing variation is modeled as:

```
t_measured = t_true + n_timing
```

Where n_timing ~ N(0, σ_timing²) is the timing measurement noise.

### Fault Injection Attacks

**Voltage Fault Model**:
Voltage glitches are modeled as temporary changes in supply voltage:

```
V_dd(t) = V_nom + A × exp(-(t-t_0)²/(2σ²)) for |t-t_0| < 3σ
```

Where:
- A is the glitch amplitude
- t_0 is the glitch timing
- σ is the glitch duration

**Success Probability**:
The probability of successful fault injection is:

```
P_fault = 1 - exp(-|A|/V_critical)
```

Where V_critical is the critical voltage for fault occurrence.

## Validation and Calibration

### Statistical Validation

All models are validated using:
1. **Goodness-of-fit tests**: Kolmogorov-Smirnov tests for distribution matching
2. **Correlation analysis**: Pearson correlation coefficients with experimental data
3. **Confidence intervals**: 95% confidence intervals for all parameter estimates

### Experimental Calibration

Model parameters are calibrated using:
1. **Published experimental data**: From peer-reviewed literature
2. **Industry benchmarks**: Standard test conditions and specifications
3. **Cross-validation**: Against independent datasets

### Uncertainty Quantification

All model predictions include uncertainty bounds based on:
1. **Parameter uncertainty**: Propagated through Monte Carlo sampling
2. **Model uncertainty**: Bayesian model averaging
3. **Measurement uncertainty**: Included in experimental comparisons

## References

### PUF Models
1. Gassend et al. "Silicon Physical Random Functions" (CCS 2002)
2. Holcomb et al. "Initial SRAM State as a Fingerprint" (HOST 2009)
3. Suh & Devadas "Physical Unclonable Functions for Device Authentication" (DAC 2007)
4. Maiti et al. "Improved Ring Oscillator PUF" (HOST 2011)

### Environmental Models
5. Sze & Ng "Physics of Semiconductor Devices" (Wiley, 2006)
6. Rabaey et al. "Digital Integrated Circuits" (Prentice Hall, 2003)
7. Schroder "Semiconductor Material and Device Characterization" (Wiley, 2006)

### Attack Models
8. Rührmair et al. "Modeling attacks on physical unclonable functions" (CCS 2010)
9. Delvaux & Verbauwhede "Side channel modeling attacks on 65nm Arbiter PUFs" (HOST 2013)
10. Tajik et al. "Physical characterization of Arbiter PUFs" (CHES 2014)

### Standards and Specifications
11. MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests
12. MIL-STD-461G: Requirements for the Control of Electromagnetic Interference Characteristics
13. IEEE 1149.1: Standard Test Access Port and Boundary-Scan Architecture
14. NIST SP 800-90B: Recommendation for the Entropy Sources Used for Random Bit Generation