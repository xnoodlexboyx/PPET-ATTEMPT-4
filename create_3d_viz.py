import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ppet.core.puf_emulator import ArbiterPUF
from ppet.core.military_stressors import MilitaryEnvironment

print('=== 3D Visualization Test ===')

# Generate 3D data for threat analysis
print('1. Generating 3D analysis data...')
temperatures = np.linspace(-40, 85, 10)
em_noise_levels = np.linspace(0, 1.0, 10)
challenge = np.random.randint(0, 2, size=64)

# Create mesh grid
T, E = np.meshgrid(temperatures, em_noise_levels)
responses = np.zeros_like(T)

# Generate response surface
for i, temp in enumerate(temperatures):
    for j, em_noise in enumerate(em_noise_levels):
        puf = ArbiterPUF(n_stages=64, environmental_stressors={
            'temperature': temp,
            'voltage': 1.2,
            'em_noise': em_noise,
            'aging_factor': 1.0
        }, seed=42)
        responses[j, i] = puf.evaluate(challenge)

print('✓ Generated 3D response surface data')

# Create 3D visualization
fig = plt.figure(figsize=(20, 12))

# 3D Surface Plot
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf = ax1.plot_surface(T, E, responses, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('EM Noise Level')
ax1.set_zlabel('Response')
ax1.set_title('3D PUF Response Surface\nTemperature vs EM Noise')

# 3D Scatter Plot - Military Environments
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
environments = [MilitaryEnvironment.GROUND_MOBILE, MilitaryEnvironment.AIRCRAFT_INTERNAL, 
                MilitaryEnvironment.AIRCRAFT_EXTERNAL, MilitaryEnvironment.NAVAL_SHELTERED,
                MilitaryEnvironment.NAVAL_EXPOSED, MilitaryEnvironment.SPACE_VEHICLE]
env_temps = []
env_noise = []
env_aging = []
env_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i, env in enumerate(environments):
    puf = ArbiterPUF(n_stages=32, military_environment=env, seed=42)
    puf.update_mission_time(500)
    env_temps.append(puf.environmental_stressors['temperature'])
    env_noise.append(puf.environmental_stressors['em_noise'])
    env_aging.append(puf.environmental_stressors['aging_factor'])
    ax2.scatter(env_temps[i], env_noise[i], env_aging[i], c=env_colors[i], s=100, alpha=0.8)

ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('EM Noise')
ax2.set_zlabel('Aging Factor')
ax2.set_title('3D Military Environment Analysis\nEnvironmental Conditions')

# Reliability over time plot
ax3 = fig.add_subplot(2, 3, 3)
times = np.linspace(0, 1000, 50)
reliability_data = []
for t in times:
    puf = ArbiterPUF(n_stages=64, military_environment=MilitaryEnvironment.GROUND_MOBILE, seed=42)
    puf.update_mission_time(t)
    # Simulate reliability as function of environmental stress
    temp_stress = abs(puf.environmental_stressors['temperature'] - 25) / 60.0  # Normalized
    em_stress = puf.environmental_stressors['em_noise']
    aging_stress = puf.environmental_stressors['aging_factor'] - 1.0
    reliability = 1.0 - (temp_stress * 0.1 + em_stress * 0.05 + aging_stress * 0.1)
    reliability_data.append(max(0.7, reliability))  # Minimum 70% reliability

ax3.plot(times, reliability_data, 'b-', linewidth=2)
ax3.set_xlabel('Mission Time (hours)')
ax3.set_ylabel('Reliability')
ax3.set_title('PUF Reliability Over Mission Time')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
ax3.legend()

# Uniqueness heatmap
ax4 = fig.add_subplot(2, 3, 4)
# Create correlation matrix between different PUFs
pufs = [ArbiterPUF(n_stages=64, seed=i) for i in range(8)]
correlation_matrix = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        if i != j:
            challenges, responses_i = pufs[i].generate_crps(100)
            _, responses_j = pufs[j].generate_crps(100, challenges)
            correlation_matrix[i, j] = np.mean(responses_i == responses_j)
        else:
            correlation_matrix[i, j] = 1.0

im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
ax4.set_xlabel('PUF Instance')
ax4.set_ylabel('PUF Instance')
ax4.set_title('PUF Instance Correlation Matrix\nResponse Similarity')
plt.colorbar(im, ax=ax4)

# Environmental stress impact
ax5 = fig.add_subplot(2, 3, 5)
stress_levels = np.linspace(0, 2, 20)
bit_error_rates = []
for stress in stress_levels:
    puf1 = ArbiterPUF(n_stages=64, environmental_stressors={'temperature': 25, 'voltage': 1.2, 'em_noise': 0, 'aging_factor': 1.0}, seed=42)
    puf2 = ArbiterPUF(n_stages=64, environmental_stressors={'temperature': 25 + stress*30, 'voltage': 1.2, 'em_noise': stress*0.5, 'aging_factor': 1.0 + stress*0.1}, seed=42)
    
    challenge = np.random.randint(0, 2, size=64)
    responses1 = [puf1.evaluate(challenge) for _ in range(50)]
    responses2 = [puf2.evaluate(challenge) for _ in range(50)]
    ber = np.mean(np.array(responses1) != np.array(responses2))
    bit_error_rates.append(ber)

ax5.plot(stress_levels, bit_error_rates, 'ro-', linewidth=2)
ax5.set_xlabel('Stress Level (normalized)')
ax5.set_ylabel('Bit Error Rate')
ax5.set_title('Environmental Stress Impact\nBit Error Rate vs Stress Level')
ax5.grid(True, alpha=0.3)

# Attack success simulation
ax6 = fig.add_subplot(2, 3, 6)
attack_types = ['ML Attack', 'Side-Channel', 'Fault Injection', 'Supply Chain', 'Brute Force']
success_rates = [0.85, 0.65, 0.45, 0.25, 0.05]  # Simulated attack success rates
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

bars = ax6.bar(attack_types, success_rates, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Success Rate')
ax6.set_title('Attack Success Rate Analysis\nDifferent Attack Types')
ax6.set_ylim(0, 1)
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{rate:.0%}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('ppet_3d_analysis.png', dpi=300, bbox_inches='tight')
print('✓ Saved 3D analysis visualization to ppet_3d_analysis.png')

print('\n=== 3D Visualization Statistics ===')
print(f'   Temperature Range: {np.min(T):.1f}°C to {np.max(T):.1f}°C')
print(f'   EM Noise Range: {np.min(E):.1f} to {np.max(E):.1f}')
print(f'   Response Variation: {np.std(responses):.3f}')
print(f'   Average Reliability: {np.mean(reliability_data):.3f}')
print(f'   Max Bit Error Rate: {max(bit_error_rates):.3f}')

print('\n=== 3D Visualization Test Complete ===')
print('✅ Advanced 3D analysis plots generated successfully!')