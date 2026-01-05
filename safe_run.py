#!/usr/bin/env python
"""
Safe wrapper to run pipeline with proper environment variables.
Prevents mutex lock errors on macOS.
"""

import os
import sys
import subprocess

# Set environment variables BEFORE any imports
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():
    """Run pipeline with safe environment."""
    print("=" * 70)
    print("Safe Pipeline Runner (macOS optimized)")
    print("=" * 70)
    print()
    
    # Get arguments (e.g., --quick, --epochs)
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--quick']
    
    print(f"Running: python run_pipeline.py {' '.join(args)}")
    print()
    print("Note: Mutex lock warnings are normal on macOS and can be ignored.")
    print("-" * 70)
    print()
    
    # Run the pipeline
    cmd = [sys.executable, 'run_pipeline.py'] + args
    
    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

