# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPET (Physical Unclonable Function Emulation and Analysis Tool) is a Python framework for emulating and analyzing Physical Unclonable Functions (PUFs) in high-security applications. The framework provides accurate PUF models, security analysis tools, and practical use cases for PUF-based authentication and secure communication in harsh environments.

## Common Development Commands

### Testing
```bash
pytest tests/
```

### Development Dependencies
Install development dependencies (linting, formatting, type checking):
```bash
pip install -e ".[dev]"
```

### Code Quality
```bash
# Formatting
black .

# Linting
flake8 .

# Type checking
mypy ppet/
```

### Installation
```bash
# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Architecture Overview

### Core Module Structure
The framework is built around a modular architecture with clear separation of concerns:

- **ppet/core/**: Core PUF simulation and analysis functionality
  - `puf_emulator.py`: Base PUF class and specific implementations (ArbiterPUF, SRAMPUF, RingOscillatorPUF)
  - `analysis.py`: PUF analysis and visualization tools with harsh environment and high-security capabilities
  - `analyzer.py`: General analysis utilities
  - `military_stressors.py`: Harsh environment modeling (temperature, EMI, aging)
  - `threat_simulator.py`: Attack simulation capabilities

- **ppet/use_cases/**: High-security application scenarios
  - `secure_communication.py`: Secure communication protocols
  - `drone_authentication.py`: Drone authentication systems

- **ppet/utilities/**: Supporting infrastructure
  - `config_manager.py`: Configuration management
  - `data_generators.py`: Test data generation
  - `logging.py`: Logging utilities

- **ppet/interfaces/**: User interaction layer
  - `cli.py`: Command-line interface (currently minimal)

### Key Design Patterns

1. **Environmental Stressor Modeling**: All PUF implementations support environmental stressors (temperature, voltage, EM noise, aging) through the `environmental_stressors` parameter and `MilitaryStressors` class.

2. **Mission Time Tracking**: PUFs can be updated with mission time to simulate degradation over operational periods using `update_mission_time()`.

3. **Hierarchical Variation Model**: PUF implementations use realistic manufacturing variation models with global, systematic, and local components.

4. **Modular Attack Simulation**: Threat simulation is designed to be extensible for different attack types.

### PUF Implementation Details

- **ArbiterPUF**: Implements delay-based PUF with path switching, environmental effects, and non-linear path interactions
- **SRAMPUF**: Models transistor mismatch in SRAM startup states with temperature and voltage dependencies
- **RingOscillatorPUF**: Frequency-based PUF with spatial correlation and process variations

### Analysis Capabilities

The `PUFAnalyzer` class provides comprehensive analysis with multiple visualization options:

#### Core Analysis Functions
- **Reliability Analysis**: Military environmental stress testing with reliability metrics over mission time
- **Uniqueness Analysis**: Pairwise Hamming distance analysis between multiple PUF instances
- **Bit-Aliasing Analysis**: Statistical analysis of bit frequency distributions across PUF instances
- **Environmental Sensitivity Analysis**: Analysis of PUF response changes under varying conditions

#### Visualization Types
- **Matplotlib Visualizations**: Traditional static plots with publication-quality output
  - Histograms of pairwise distances and aliasing deviations
  - Scatterplots for uniqueness analysis
  - Box plots for distribution analysis
  - Line graphs for time-series data
  - Heatmaps for correlation matrices and 2D data
- **Plotly Visualizations**: Interactive plots with zoom, pan, and hover capabilities
  - Interactive 3D plots for multi-dimensional analysis
  - Interactive heatmaps and scatter plots
  - HTML output for web-based reporting

#### Advanced 3D Visualization Functions
- **3D Threat Landscape**: `plot_3d_threat_landscape()` - Interactive 3D surface plots showing attack success rates vs environmental conditions
- **3D PUF Response Analysis**: `plot_3d_puf_response_analysis()` - Challenge complexity vs response vs reliability visualization
- **3D Environmental Stress Impact**: `plot_3d_environmental_stress_impact()` - Temperature/EMI/time correlation analysis
- **3D Multi-Attack Comparison**: `plot_3d_multi_attack_comparison()` - Comparative attack effectiveness visualization

#### Comprehensive Reporting
- `generate_comprehensive_report()`: Automated generation of all analysis metrics and visualizations
- Support for both Matplotlib (PNG) and Plotly (HTML) output formats
- Batch processing of multiple PUF instances
- Military-grade metrics and assessment

#### Threat Assessment Module
- `ThreatAssessmentReportGenerator`: Complete defense procurement report generation
- Automated risk assessment and classification
- Defense-specific countermeasure recommendations
- Executive summary generation for decision makers
- HTML and JSON report formats

## Configuration

The framework supports YAML/JSON configuration files for:
- PUF parameters (challenge length, noise levels, variation parameters)
- Authentication thresholds
- Logging configuration
- Environmental stressor profiles

## Important Implementation Notes

1. **Defensive Security Focus**: This framework is designed for defensive security research and analysis only.

2. **Realistic Modeling**: Environmental effects are modeled based on actual physical phenomena with realistic parameter ranges.

3. **Extensibility**: New PUF types should inherit from the base `PUF` class and implement the required methods (`generate_crps`, `evaluate`).

4. **Industrial Standards**: Analysis tools are designed to meet industrial reliability and security assessment requirements.

5. **Reproducibility**: All simulations support seeding for reproducible results.

## Data Flow

1. **Configuration**: Load PUF parameters and environmental conditions
2. **Simulation**: Generate PUF instances with realistic variations
3. **Analysis**: Evaluate metrics (uniqueness, reliability, security)
4. **Visualization**: Generate plots and reports
5. **Output**: Results displayed via interface or saved to files

## Dependencies

Core dependencies:
- `numpy`: Numerical computations and array operations
- `scipy`: Signal processing and statistical functions
- `scikit-learn`: Machine learning for attack simulation
- `matplotlib`/`seaborn`: Static visualization and statistical plots
- `plotly`: Interactive visualization and 3D plotting
- `pandas`: Data manipulation
- `pyyaml`: Configuration file parsing
- `pytest`: Testing framework

Development dependencies:
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking
- `pytest-cov`: Coverage testing
- `sphinx`: Documentation generation

## Examples and Usage

### Basic Analysis Example
```python
from ppet.core.puf_emulator import ArbiterPUF
from ppet.core.analysis import PUFAnalyzer

# Create multiple PUF instances
pufs = [ArbiterPUF(n_stages=64, seed=i) for i in range(10)]
analyzer = PUFAnalyzer(pufs[0])

# Analyze uniqueness
uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=1000)
analyzer.plot_uniqueness_analysis(uniqueness_data, use_plotly=True)

# Analyze bit-aliasing
aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=1000)
analyzer.plot_bit_aliasing_analysis(aliasing_data, use_plotly=False)
```

### Advanced Examples

#### Comprehensive Analysis Example
See `examples/comprehensive_analysis_example.py` for a complete demonstration of all analysis capabilities including:
- Uniqueness analysis with multiple visualization types
- Bit-aliasing analysis with heatmaps and bar graphs
- Environmental stress testing
- Comprehensive report generation

#### 3D Defense Visualization Example
See `examples/defense_3d_visualization_example.py` for defense-specific 3D visualizations including:
- 3D threat landscape analysis for multiple attack types
- Interactive environmental stress impact visualization
- Multi-attack comparison with 3D scatter plots
- Satellite communication stress testing
- Battlefield IoT device analysis
- Drone swarm authentication analysis

#### Threat Assessment Example
See `examples/threat_assessment_example.py` for complete defense procurement reporting including:
- Ground mobile system assessment
- Aircraft system assessment  
- Space vehicle system assessment
- Naval system assessment
- Comparative analysis across environments
- Executive summary generation

### Defense Use Case Examples

#### Satellite Communication Analysis
```python
from ppet.core.threat_assessment import ThreatAssessmentReportGenerator
from ppet.core.military_stressors import MilitaryEnvironment

# Create space-hardened PUFs
pufs = [ArbiterPUF(n_stages=256, seed=i) for i in range(8)]

# Generate comprehensive threat assessment
report_gen = ThreatAssessmentReportGenerator(classification_level="TOP_SECRET")
report_data = report_gen.generate_defense_procurement_report(
    pufs=pufs,
    environment=MilitaryEnvironment.SPACE_VEHICLE,
    use_cases=["Satellite Authentication", "Secure Telemetry", "Command Authorization"],
    output_dir="space_vehicle_threat_assessment"
)
```

#### Battlefield IoT Device Analysis
```python
# Create IoT-optimized PUFs
iot_pufs = [ArbiterPUF(n_stages=64, seed=i) for i in range(10)]

# Analyze uniqueness and 3D response patterns
analyzer = PUFAnalyzer(iot_pufs[0])
uniqueness_data = analyzer.analyze_uniqueness(iot_pufs)
analyzer.plot_3d_puf_response_analysis(iot_pufs, save_path="iot_3d_analysis.html")
```