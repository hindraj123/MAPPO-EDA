"""Static checks for the Linux-only official ORFS benchmark setup.

These tests do not execute OpenROAD and do not report physical-design metrics.
Real verification remains pending on Ubuntu with the pinned ORFS checkout.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_openroad_config_pins_orfs_and_official_gcd_paths():
    text = (ROOT / "configs/openroad.yaml").read_text(encoding="utf-8")
    assert "orfs_revision: 8b8d340" in text
    assert "flow/designs/nangate45/gcd/config.mk" in text
    assert "flow/results/nangate45/gcd/base/6_final.gds" in text
    assert "real_verification: pending" in text


def test_linux_setup_does_not_install_or_require_verilator():
    text = (ROOT / "scripts/setup_linux.sh").read_text(encoding="utf-8")
    assert "sudo" in text  # documented administrator commands only
    assert "apt-get install" in text
    assert "verilator" not in text.lower()
    assert "does not run sudo" in text


def test_orfs_verification_script_is_pinned_and_non_vendoring():
    text = (ROOT / "scripts/openroad/verify_orfs_gcd.sh").read_text(encoding="utf-8")
    assert "expected_revision=\"8b8d340\"" in text
    assert "DESIGN_CONFIG=./designs/nangate45/gcd/config.mk" in text
    assert "6_final.gds" in text
    assert "vendor" in text.lower()


def test_benchmark_does_not_include_technology_files():
    benchmark = ROOT / "benchmarks"
    forbidden_suffixes = {".lef", ".lib", ".gds", ".map", ".techlef"}
    assert not [path for path in benchmark.rglob("*") if path.suffix.lower() in forbidden_suffixes]
