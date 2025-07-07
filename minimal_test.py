#!/usr/bin/env python3
"""Minimal test to isolate import issues."""

import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing step by step...")
    
    print("1. Testing operators...")
    from epg_python.operators import rf_rotation, shift_grad, relax
    print("   ✓ Operators OK")
    
    print("2. Testing utils...")
    from epg_python.utils import get_relaxation_params
    print("   ✓ Utils OK")
    
    print("3. Testing core...")
    from epg_python.core import EPGSequence, EPGSimulator, epg_custom
    print("   ✓ Core OK")
    
    print("4. Testing sequences...")
    from epg_python.sequences import EPGSequenceTSE, EPGSequenceGRE, EPGSequenceVFA
    print("   ✓ Sequence classes OK")
    
    print("5. Testing sequence functions...")
    from epg_python.sequences import simulate_tse, simulate_gre, simulate_vfa
    print("   ✓ Sequence functions OK")
    
    print("6. Testing visualization...")
    from epg_python.visualization import display_epg, plot_echoes
    print("   ✓ Visualization OK")
    
    print("7. Testing main package...")
    from epg_python import EPGSimulator, simulate_tse
    print("   ✓ Main package OK")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()