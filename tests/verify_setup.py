#!/usr/bin/env python3
"""Quick verification script to check test setup.

Run this to verify that the test suite is properly configured:
    python tests/verify_setup.py
"""
import sys
from pathlib import Path


def verify_test_structure():
    """Verify that all test directories and key files exist."""
    print("Checking test suite structure...")

    required_dirs = [
        "tests",
        "tests/test_tiling",
        "tests/test_mvt",
        "tests/test_histogram",
        "tests/test_stats",
        "tests/test_server",
    ]

    required_files = [
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/README.md",
        "pytest.ini",
        "tests/test_tiling/test_rsgrove.py",
        "tests/test_mvt/test_helpers.py",
        "tests/test_mvt/test_assigner.py",
        "tests/test_stats/test_sketches.py",
    ]

    errors = []

    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            errors.append(f"Missing directory: {dir_path}")
        else:
            print(f"  ✓ {dir_path}")

    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            errors.append(f"Missing file: {file_path}")
        else:
            print(f"  ✓ {file_path}")

    return errors


def check_imports():
    """Check that pytest and other test dependencies can be imported."""
    print("\nChecking test dependencies...")

    try:
        import pytest
        print(f"  ✓ pytest {pytest.__version__}")
    except ImportError:
        return ["pytest not installed. Run: pip install -e '.[dev]'"]

    try:
        import pytest_cov
        print("  ✓ pytest-cov")
    except ImportError:
        return ["pytest-cov not installed. Run: pip install -e '.[dev]'"]

    try:
        import pytest_mock
        print("  ✓ pytest-mock")
    except ImportError:
        return ["pytest-mock not installed. Run: pip install -e '.[dev]'"]

    return []


def check_starlet_imports():
    """Check that starlet modules can be imported."""
    print("\nChecking starlet imports...")

    modules_to_check = [
        "starlet._internal.tiling.RSGrove",
        "starlet._internal.mvt.helpers",
        "starlet._internal.mvt.assigner",
        "starlet._internal.stats.sketches",
    ]

    errors = []

    for module in modules_to_check:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            errors.append(f"Cannot import {module}: {e}")

    return errors


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Starlet Test Suite Verification")
    print("=" * 60)

    all_errors = []

    # Check test structure
    all_errors.extend(verify_test_structure())

    # Check dependencies
    all_errors.extend(check_imports())

    # Check starlet imports
    all_errors.extend(check_starlet_imports())

    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print("FAILED: Issues found")
        print("=" * 60)
        for error in all_errors:
            print(f"  ✗ {error}")
        return 1
    else:
        print("SUCCESS: Test suite is properly configured!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Install dev dependencies: pip install -e '.[dev]'")
        print("  2. Run tests: pytest")
        print("  3. Run with coverage: pytest --cov=starlet --cov-report=html")
        print("  4. Skip slow tests: pytest -m 'not slow'")
        return 0


if __name__ == "__main__":
    sys.exit(main())
