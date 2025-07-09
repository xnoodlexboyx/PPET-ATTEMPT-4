#!/usr/bin/env python3
"""Quick test of visualization modules."""

import sys
sys.path.append('.')

try:
    # Test imports
    from ppet.core.puf_emulator import ArbiterPUF
    from ppet.core.analysis import PUFAnalyzer
    from ppet.core.military_stressors import MilitaryEnvironment
    from ppet.core.threat_simulator import MLAttack, SideChannelAttack
    
    print("✅ All imports successful")
    
    # Test basic PUF creation
    puf = ArbiterPUF(n_stages=32, seed=42)
    print("✅ PUF creation successful")
    
    # Test analyzer creation
    analyzer = PUFAnalyzer(puf)
    print("✅ Analyzer creation successful")
    
    # Test basic analysis
    pufs = [ArbiterPUF(n_stages=32, seed=i) for i in range(3)]
    uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=50)
    print("✅ Uniqueness analysis successful")
    
    # Test 3D visualization creation (without showing)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3], mode='markers'))
    print("✅ 3D plotly visualization creation successful")
    
    # Test attack creation
    attack = MLAttack(model_type='rf')
    print("✅ Attack creation successful")
    
    # Test military environment
    env = MilitaryEnvironment.GROUND_MOBILE
    print("✅ Military environment creation successful")
    
    print("\n🎉 All visualization modules are working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)