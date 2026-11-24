import json, sys, tarfile, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_tool
import pytest


def make_bundle(path: Path, members: dict, env: dict):
    with tarfile.open(path, "w:gz") as t:
        for name, data in {**members, "env.json": json.dumps(env).encode()}.items():
            ti = tarfile.TarInfo(name); ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))


def test_extract_rejects_path_escape(tmp_path):
    b = tmp_path / "b.tar.gz"
    make_bundle(b, {"../evil.txt": b"x"}, {})
    with pytest.raises(SystemExit):
        run_tool.extract(b, tmp_path / "root")


def test_resolve_replaces_step_output_refs(tmp_path):
    outs = [tmp_path / "step0"]
    ref = "$" + "{out:0}/pose.png"
    got = run_tool.resolve({"image": ref, "n": [ref, 3]}, tmp_path, outs)
    assert got == {"image": f"{outs[0]}/pose.png", "n": [f"{outs[0]}/pose.png", 3]}


def test_run_step_runs_the_tool_argv_unchanged_with_remapped_paths(tmp_path):
    root = tmp_path / "hq"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "t.py").write_text(
        "import json,sys,pathlib\n"
        "p=json.load(open(sys.argv[2])); out=pathlib.Path(sys.argv[4])\n"
        "(out/'r.txt').write_text(p['x'])\n")
    step = {"tool": "demo.t", "params": {"x": "hello"},
            "runner": {"argv": ["python3", "t.py", "--params", "{params_json}", "--out", "{out_dir}"], "cwd": "/home/user/hq/tools"}}
    outs = []
    res = run_tool.run_step(0, step, root, outs)
    assert res["rc"] == 0
    assert (outs[0] / "r.txt").read_text() == "hello"
