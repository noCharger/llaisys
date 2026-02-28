import os
import sys
import argparse
import subprocess
import glob

def main():
    parser = argparse.ArgumentParser(description="Run all operator tests")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run tests on (cpu/nvidia)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling for operator tests")
    args = parser.parse_args()

    # Get directory of current script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ops_dir = os.path.join(test_dir, "ops")

    # Find all python files in test/ops/
    test_files = glob.glob(os.path.join(ops_dir, "*.py"))
    # Filter out __init__.py and other non-test files
    test_files = [f for f in test_files if os.path.basename(f).endswith(".py") and not os.path.basename(f).startswith("__")]
    test_files.sort()

    print(f"Found {len(test_files)} operator tests in {ops_dir}")
    print(f"Target Device: {args.device}\n")

    failed_tests = []
    passed_tests = []

    for test_file in test_files:
        test_name = os.path.basename(test_file)
        print(f"Running {test_name}...", end=" ", flush=True)
        
        # Run the test script as a subprocess
        cmd = [sys.executable, test_file, "--device", args.device]
        if args.profile:
            cmd.append("--profile")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("\033[92mPASSED\033[0m")
            passed_tests.append(test_name)
        else:
            print("\033[91mFAILED\033[0m")
            print(f"Output for {test_name}:\n{result.stdout}")
            print(f"Error output for {test_name}:\n{result.stderr}")
            failed_tests.append(test_name)

    print("\n" + "="*40)
    print(f"Summary: {len(passed_tests)} PASSED, {len(failed_tests)} FAILED")
    print("="*40)

    if failed_tests:
        print("Failed tests:")
        for t in failed_tests:
            print(f"  - {t}")
        sys.exit(1)
    else:
        print("All operator tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
