import json
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.environ.get("SOLUTION_DIR", "."))

SOLUTION_DIR = os.environ.get("SOLUTION_DIR", ".")


def _run_in_subprocess():
    """Import and call the solution in a clean process, capturing stderr.

    MLX prints its deprecation notices to stderr from C++ rather than raising a
    Python DeprecationWarning, so warnings.catch_warnings cannot see them. Only a
    subprocess with captured stderr can.
    """
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        import solution
        r = solution.memory_report()
        print("REPORT_JSON:" + json.dumps({k: (v if isinstance(v, (int, str)) else str(v)) for k, v in r.items()}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, SOLUTION_DIR],
        capture_output=True, text=True, timeout=300,
    )
    return proc


def test_returns_expected_keys_and_types():
    proc = _run_in_subprocess()
    assert proc.returncode == 0, f"solution failed to run:\n{proc.stderr[-800:]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("REPORT_JSON:")]
    assert line, f"no report produced; stdout was:\n{proc.stdout[-500:]}"
    d = json.loads(line[0][len("REPORT_JSON:") :])

    assert set(d) == {"active_bytes", "peak_bytes", "cache_bytes", "device_name"}, (
        f"wrong keys: {sorted(d)}"
    )
    for k in ("active_bytes", "peak_bytes", "cache_bytes"):
        assert isinstance(d[k], int), f"{k} should be int, got {type(d[k]).__name__}"
        assert d[k] >= 0, f"{k} is negative"
    assert isinstance(d["device_name"], str) and d["device_name"], "device_name empty"


def test_no_deprecation_notice():
    proc = _run_in_subprocess()
    stderr = proc.stderr.lower()
    assert "deprecated" not in stderr, (
        "solution used a deprecated MLX API; stderr contained:\n"
        + "\n".join(l for l in proc.stderr.splitlines() if "deprecated" in l.lower())
    )


def test_device_name_is_plausible():
    proc = _run_in_subprocess()
    line = [x for x in proc.stdout.splitlines() if x.startswith("REPORT_JSON:")][0]
    d = json.loads(line[len("REPORT_JSON:") :])
    assert "Apple" in d["device_name"], f"unexpected device name {d['device_name']!r}"
