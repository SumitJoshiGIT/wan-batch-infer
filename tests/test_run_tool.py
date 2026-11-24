import json, sys, tarfile, io, types
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

def test_resolve_makes_existing_relative_path_absolute(tmp_path):
    (tmp_path / "out" / "cond").mkdir(parents=True)
    (tmp_path / "out" / "cond" / "pose0.png").write_bytes(b"x")
    got = run_tool.resolve({"image": "out/cond/pose0.png"}, tmp_path, [])
    assert got == {"image": str((tmp_path / "out/cond/pose0.png").resolve())}
    assert Path(got["image"]).is_absolute()


def test_resolve_makes_existing_relative_dir_absolute(tmp_path):
    (tmp_path / "out").mkdir()
    assert run_tool.resolve("out", tmp_path, []) == str((tmp_path / "out").resolve())


def test_resolve_leaves_nonexistent_strings_numbers_and_urls_untouched(tmp_path):
    (tmp_path / "out").mkdir()
    for s in ["a prompt, with spaces", "out/missing.png", "https://example.com/a.png", "", "cat"]:
        assert run_tool.resolve(s, tmp_path, []) == s
    assert run_tool.resolve(3, tmp_path, []) == 3
    assert run_tool.resolve(2.5, tmp_path, []) == 2.5
    assert run_tool.resolve(None, tmp_path, []) is None


def test_resolve_keeps_absolute_paths_and_out_refs(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert run_tool.resolve(str(f), tmp_path, []) == str(f)
    outs = [tmp_path / "step0"]
    assert run_tool.resolve("${out:0}", tmp_path, outs) == str(outs[0])


def test_resolve_recurses_into_lists_and_dicts(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "m.png").write_bytes(b"x")
    val = {"a": ["out/m.png", 1, {"deep": ["out/m.png", "keep me"]}], "b": {"c": "out/m.png"}}
    want = str((tmp_path / "out/m.png").resolve())
    got = run_tool.resolve(val, tmp_path, [])
    assert got == {"a": [want, 1, {"deep": [want, "keep me"]}], "b": {"c": want}}


def test_dep_failed_true_when_referenced_step_failed():
    results = [{"i": 0, "rc": 1}]
    assert run_tool.dep_failed({"x": "${out:0}/a.png"}, results) is True
    assert run_tool.failed_deps({"x": ["${out:0}"]}, results) == [0]


def test_dep_failed_false_when_referenced_step_passed():
    results = [{"i": 0, "rc": 0}]
    assert run_tool.dep_failed({"x": "${out:0}/a.png"}, results) is False
    assert run_tool.dep_failed({"x": "no refs here"}, results) is False


def test_dep_failed_true_when_referenced_step_was_skipped():
    results = [{"i": 0, "rc": 1}, run_tool.skipped_record(1, {"tool": "t.a"}, 0)]
    assert run_tool.dep_failed({"x": "${out:1}"}, results) is True
    assert run_tool.failed_deps({"x": "${out:0} ${out:1}"}, results) == [0, 1]


def test_dep_failed_raises_on_reference_to_a_step_not_yet_run():
    with pytest.raises(ValueError):
        run_tool.dep_failed({"x": "${out:2}"}, [{"i": 0, "rc": 0}])


def test_skipped_record_shape():
    rec = run_tool.skipped_record(3, {"tool": "demo.t", "params": {}}, 1)
    assert rec == {"i": 3, "tool": "demo.t", "rc": None, "skipped": True, "reason": "dependency step 1 failed"}


class _FakeOp:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeHub:
    instances = []

    def __init__(self, token=None):
        self.commits = []
        _FakeHub.instances.append(self)

    def create_commit(self, dataset, ops, msg):
        self.commits.append((ops, msg))


def test_main_skips_dependents_of_failed_step_and_runs_independent_ones(tmp_path, monkeypatch):
    env = {"steps": [
        {"tool": "demo.a", "params": {}, "runner": {}},                          # 0: fails
        {"tool": "demo.b", "params": {"img": "${out:0}/x.png"}, "runner": {}},  # 1: depends on 0
        {"tool": "demo.c", "params": {"n": 1}, "runner": {}},                    # 2: independent, passes
        {"tool": "demo.d", "params": {"img": "${out:1}"}, "runner": {}},        # 3: depends on skipped 1
    ]}
    hub_mod = types.ModuleType("hub")
    hub_mod.HFHub = _FakeHub
    hub_mod.CommitOperation = _FakeOp
    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.hf_hub_download = lambda *a, **k: str(tmp_path / "bundle.tar.gz")
    monkeypatch.setitem(sys.modules, "hub", hub_mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
    monkeypatch.setitem(sys.modules, "comfy_setup", types.ModuleType("comfy_setup"))
    monkeypatch.setattr(_FakeHub, "instances", [])
    monkeypatch.setattr(run_tool, "pick_root", lambda: tmp_path / "root")
    monkeypatch.setattr(run_tool, "extract", lambda b, r: env)
    monkeypatch.setattr(run_tool, "setup_comfy", lambda r, e: None)
    monkeypatch.setattr(run_tool, "pip_install", lambda e, c: None)
    monkeypatch.setattr(run_tool, "fetch_models", lambda r, e, t: [])
    monkeypatch.setattr(run_tool, "beat", lambda *a, **k: None)
    monkeypatch.setattr(run_tool, "upload_dir", lambda *a, **k: [])
    ran = []

    def fake_run_step(i, step, root, outs):
        ran.append(i)
        out = tmp_path / f"out{i}"
        out.mkdir(exist_ok=True)
        outs.append(out)
        return {"i": i, "tool": step["tool"], "rc": 1 if i == 0 else 0, "sec": 0, "out_dir": str(out)}

    monkeypatch.setattr(run_tool, "run_step", fake_run_step)
    rc = run_tool.main(["--bundle", "b", "--job-dataset", "j", "--hub-dataset", "h", "--run-id", "r", "--keep-going"])
    assert rc == 1
    assert ran == [0, 2]  # 1 and 3 never run; the independent step 2 still runs
    ops, _ = _FakeHub.instances[-1].commits[-1]  # the DONE.json commit is the last one
    done = json.loads(ops[0].content_bytes)
    assert done["ok"] is False
    assert [s["i"] for s in done["steps"]] == [0, 1, 2, 3]
    assert done["steps"][1] == {"i": 1, "tool": "demo.b", "rc": None, "skipped": True, "reason": "dependency step 0 failed"}
    assert done["steps"][2]["rc"] == 0 and "skipped" not in done["steps"][2]
    assert done["steps"][3] == {"i": 3, "tool": "demo.d", "rc": None, "skipped": True, "reason": "dependency step 1 failed"}

