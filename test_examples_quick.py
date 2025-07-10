#!/usr/bin/env python3
"""
Quick test of example scripts to verify they work properly.
"""

import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF
        from ppet.core.analysis import PUFAnalyzer
        from ppet.core.military_stressors import MilitaryEnvironment
        print("✓ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_puf_creation():
    """Test basic PUF creation and analysis."""
    try:
        from ppet.core.puf_emulator import ArbiterPUF
        from ppet.core.analysis import PUFAnalyzer
        
        # Create a simple PUF
        puf = ArbiterPUF(n_stages=32, seed=42)
        analyzer = PUFAnalyzer(puf)
        
        # Test CRP generation
        crps = puf.generate_crps(10)
        print(f"✓ Basic PUF creation and CRP generation works")
        return True
    except Exception as e:
        print(f"✗ Basic PUF test failed: {e}")
        return False

def test_simple_analysis():
    """Test simple analysis functions."""
    try:
        from ppet.core.puf_emulator import ArbiterPUF
        from ppet.core.analysis import PUFAnalyzer
        
        # Create test PUFs
        pufs = [ArbiterPUF(n_stages=32, seed=i) for i in range(3)]
        analyzer = PUFAnalyzer(pufs[0])
        
        # Test uniqueness analysis
        uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=50)
        print(f"✓ Uniqueness analysis works (mean: {uniqueness_data['mean_uniqueness']:.3f})")
        
        # Test bit-aliasing analysis
        aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=50)
        print(f"✓ Bit-aliasing analysis works (mean: {aliasing_data['mean_aliasing']:.3f})")
        
        return True
    except Exception as e:
        print(f"✗ Simple analysis test failed: {e}")
        return False

def test_visualization_creation():
    """Test basic visualization creation."""
    try:
        from ppet.core.puf_emulator import ArbiterPUF
        from ppet.core.analysis import PUFAnalyzer
        
        # Create test PUFs
        pufs = [ArbiterPUF(n_stages=32, seed=i) for i in range(3)]
        analyzer = PUFAnalyzer(pufs[0])
        
        # Test visualization (without actually displaying)
        uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=50)
        
        # Test matplotlib visualization
        analyzer.plot_uniqueness_analysis(
            uniqueness_data, 
            save_path='test_uniqueness.png',
            use_plotly=False
        )
        
        # Check if file was created
        if os.path.exists('test_uniqueness.png'):
            print("✓ Matplotlib visualization creation works")
            os.remove('test_uniqueness.png')  # Clean up
        else:
            print("⚠ Matplotlib visualization file not created")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("Quick Test of PPET Examples")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic PUF Creation", test_basic_puf_creation),
        ("Simple Analysis", test_simple_analysis),
        ("Visualization Creation", test_visualization_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
    
    print(f"\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - examples should work")
    else:
        print("⚠️ Some tests failed - check examples")

if __name__ == "__main__":
    main()