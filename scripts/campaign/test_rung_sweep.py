import argparse
import array
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import campaign
import rung_env
import rung_sweep

SPEC = '''
const PREFILL_RUNGS: &[&str] = &["prefill"];
    KnobSpec::new("emit.no_glu_fuse", Some("PLOW_NO_GLU_FUSE"), Layer::Emit, Domain::Bool, LT, Q).per_rung(PREFILL_RUNGS),
    KnobSpec::new("emit.fp8_kv", Some("PLOW_FP8_KV"), Layer::Emit, Domain::Bool, OFF, OPT_IN),
    KnobSpec::new("emit.both", Some("PLOW_BOTH"), Layer::Emit, Domain::Bool, OFF, OPT_IN).per_rung(&["prefill", "decode"]),
'''


class RungEnvTests(unittest.TestCase):
    def test_registry_reads_per_rung_declarations(self):
        emit, rungs = rung_env.registry(SPEC)
        self.assertEqual(emit["PLOW_FP8_KV"], "emit.fp8_kv")
        self.assertEqual(rungs, {"PLOW_NO_GLU_FUSE": ("prefill",), "PLOW_BOTH": ("prefill", "decode")})
        # The real registry: NO_GLU_FUSE is the first per-rung knob; QNORM stays packet-wide.
        emit, rungs = rung_env.registry()
        self.assertEqual(rungs["PLOW_NO_GLU_FUSE"], ("prefill",))
        self.assertNotIn("PLOW_QNORM_FUSE", rungs)

    def test_recipe_string_round_trip_and_canonical_order(self):
        recipe = {"emit": {"rung_env": {"prefill": {"4096": {"PLOW_NO_GLU_FUSE": "0"},
                                                    "128": {"PLOW_NO_GLU_FUSE": "1"}}}}}
        over = rung_env.from_recipe(recipe)
        s = rung_env.render(over)
        self.assertEqual(s, "prefill:128:PLOW_NO_GLU_FUSE=1;prefill:4096:PLOW_NO_GLU_FUSE=0")
        self.assertEqual(rung_env.parse(s), over)
        merged = rung_env.merge(over, rung_env.parse("prefill:128:PLOW_NO_GLU_FUSE=0;prefill:512:PLOW_NO_GLU_FUSE=0"))
        self.assertEqual(rung_env.render(merged), "prefill:128:PLOW_NO_GLU_FUSE=0;prefill:512:PLOW_NO_GLU_FUSE=0;"
                                                  "prefill:4096:PLOW_NO_GLU_FUSE=0")
        self.assertEqual(rung_env.check(over), [])
        self.assertIn("[emit.rung_env.prefill.128]", rung_env.toml_snippet(over))

    def test_refusals(self):
        bad = {("prefill", 128): {"PLOW_FP8_KV": "1"}, ("decode", 8): {"PLOW_NO_GLU_FUSE": "1"},
               ("tail", 8): {"PLOW_BOTH": "1"}, ("decode", 0): {"PLOW_BOTH": "1"},
               ("prefill", 256): {"PLOW_INVENTED": "1", "PLOW_NO_GLU_FUSE": "a;b"}}
        errs = "\n".join(rung_env.check(bad, SPEC))
        for why in ("PLOW_FP8_KV: emit.fp8_kv is packet-wide", "decode:8:PLOW_NO_GLU_FUSE: emit.no_glu_fuse is packet-wide",
                    "tail:8: phase", "decode:0: rows", "PLOW_INVENTED: not a registered", "without ; : ="):
            self.assertIn(why, errs)
        for spec in ("prefill:128:PLOW_X", "prefill:x:PLOW_X=1", "prefill:128:PLOW_X=1;prefill:128:PLOW_X=0"):
            with self.assertRaises(rung_env.RungEnvError):
                rung_env.parse(spec)
        for table in ({"prefill": {"0": {"PLOW_X": "1"}}}, {"prefill": {"128": {"PLOW_X": 1}}}, {"prefill": "x"}):
            with self.assertRaises(rung_env.RungEnvError):
                rung_env.from_recipe({"emit": {"rung_env": table}})
        self.assertEqual([rung_env.boolish(v) for v in ("1", "On", "0", "false", "x")], ["true", "true", "false", "false", None])


def write_build(root: Path, name: str, value: str, programs: dict[int, list]) -> Path:
    """A campaign build dir whose build.json carries what rung_sweep reads."""
    d = root / name
    (d / "assets" / "objects").mkdir(parents=True)
    rows = sorted(programs)
    bj = {
        "shapes": {"prefill_buckets": rows, "max_chunk": 8192, "decode_batch": 128},
        "emit_config": {"knobs": [{"id": "max_request_chunk", "env": "PLOW_MAX_REQUEST_CHUNK", "value": "4096"},
                                  {"id": "no_glu_fuse", "env": "PLOW_NO_GLU_FUSE", "value": value}]},
        "rung_knobs": {"programs": [{"program": i, "phase": "prefill", "rows": r, "packed": False,
                                     "knobs": {"emit.no_glu_fuse": {"value": value, "source": "rung"}}}
                                    for i, r in enumerate(rows)]},
        "kernel_cases": {"programs": [{"program": i, "kind": "prefill", "rows": r, "packed_only": False,
                                       "cases": programs[r]} for i, r in enumerate(rows)]},
        "dispatch_chains": [], "programs": [{"kind": "prefill", "bucket": r, "arms": [str(programs[r][0])]} for r in rows],
    }
    (d / "assets" / "build.json").write_text(json.dumps(bj))
    return d


def write_run(runs: Path, tag: str, rung: dict, ms: list[float], logits: list[float] | None, rc: int = 0) -> None:
    lines = [f"packed prefill: rep={i} prefix=0 chunk_rows={rung['ctx']} requests={rung['slots']} "
             f"rows={rung['rows']} ms={m:.3f}" for i, m in enumerate(ms)]
    (runs / f"{tag}.log").write_text("\n".join(lines + [f"rc={rc}"]) + "\n")
    if logits is None:
        return
    d = runs / f"{tag}.logits"
    d.mkdir()
    bf16 = array.array("f", logits).tobytes()
    raw = bytearray(len(bf16) // 2)
    raw[0::2], raw[1::2] = bf16[2::4], bf16[3::4]
    (d / "prefill-s000.bin").write_bytes(bytes(raw))
    (d / "manifest.json").write_text(json.dumps({"cases": [{"id": "prefill-s000", "file": "prefill-s000.bin",
                                                            "prompt_sha256_u32le": "p"}]}))


class RungSweepTests(unittest.TestCase):
    def test_plan_fits_one_launch_per_rung(self):
        bj = {"shapes": {"prefill_buckets": [128, 4096, 8192], "max_chunk": 8192, "decode_batch": 128},
              "emit_config": {"knobs": [{"id": "max_request_chunk", "value": "4096"}]}}
        self.assertEqual(rung_sweep.plan_rungs(bj, [128, 8192]),
                         [{"rows": 128, "slots": 1, "ctx": 128}, {"rows": 8192, "slots": 2, "ctx": 4096}])
        with self.assertRaises(ValueError):
            rung_sweep.plan_rungs(bj, [96])
        self.assertEqual(rung_sweep.block_orders(["a", "b"], 2), [["a", "b"], ["b", "a"]])

    def test_knob_must_be_rung_eligible(self):
        self.assertEqual(rung_sweep.knob_id("PLOW_NO_GLU_FUSE"), "emit.no_glu_fuse")
        with self.assertRaises(ValueError):
            rung_sweep.knob_id("PLOW_QNORM_FUSE")

    def test_run_script_interleaves_arms(self):
        plan = {"out": "/x", "rungs": [{"rows": 128, "slots": 1, "ctx": 128}], "arms": ["v0", "v1"], "blocks": 2,
                "builds": {"v0": "/b0", "v1": "/b1"}, "reps": 6, "timeout_s": 600}
        s = rung_sweep.run_script(plan, Path("/t/step_bench"), {"PLOW_LIVE_CTX": "16384"})
        runs = [ln for ln in s.splitlines() if ln.startswith("run ")]
        self.assertEqual([ln.split()[1] for ln in runs], ["r128-b0-v0", "r128-b0-v1", "r128-b1-v1", "r128-b1-v0"])
        self.assertIn("--packed-prefill --pf-reps 6 --dump-prefill-logits", s)
        self.assertIn("export PLOW_LIVE_CTX=16384", s)

    def test_parse_log_refuses_a_different_width(self):
        with tempfile.TemporaryDirectory() as t:
            runs = Path(t)
            rung = {"rows": 8192, "slots": 2, "ctx": 4096}
            write_run(runs, "ok", rung, [90.0, 81.0, 80.0, 82.0], None)
            p = rung_sweep.parse_log(runs / "ok.log", rung, 4)
            self.assertTrue(p["ok"], p)
            self.assertEqual(p["ms"], 81.0)
            self.assertFalse(rung_sweep.parse_log(runs / "ok.log", rung, 5)["ok"], "a missing rep")
            write_run(runs, "narrow", {"rows": 4096, "slots": 1, "ctx": 4096}, [50.0, 40.0], None)
            self.assertFalse(rung_sweep.parse_log(runs / "narrow.log", rung, 2)["ok"])
            write_run(runs, "crash", rung, [90.0, 80.0], None, rc=134)
            self.assertIn("rc=134", rung_sweep.parse_log(runs / "crash.log", rung, 2)["why"])

    def test_score_picks_per_rung_winners_and_writes_the_snippet(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            runs = root / "runs"
            runs.mkdir()
            fused, unfused = {128: [20, 1], 1024: [20, 2]}, {128: [10, 1], 1024: [10, 2]}
            ref = write_build(root, "ref", "true", unfused)
            arms = {"v0": write_build(root, "v0", "false", fused), "v1": write_build(root, "v1", "true", unfused)}
            plan = {"knob": "PLOW_NO_GLU_FUSE", "values": ["0", "1"], "arms": ["v0", "v1"], "blocks": 2, "reps": 4,
                    "rungs": [{"rows": 128, "slots": 1, "ctx": 128}, {"rows": 1024, "slots": 1, "ctx": 1024}]}
            base = [0.5 * i for i in range(256)]
            near = [x + (0.01 if i == 3 else 0.0) for i, x in enumerate(base)]
            # 128: fused (v0) clearly faster, logits differ slightly. 1024: unfused (v1) faster.
            timings = {(128, "v0"): [30.0, 15.0, 15.1, 15.0], (128, "v1"): [30.0, 17.2, 17.3, 17.2],
                       (1024, "v0"): [40.0, 30.8, 30.9, 30.8], (1024, "v1"): [40.0, 26.5, 26.4, 26.5]}
            for (rows, arm), ms in timings.items():
                rung = next(r for r in plan["rungs"] if r["rows"] == rows)
                for b in range(2):
                    write_run(runs, f"r{rows}-b{b}-{arm}", rung, ms, near if arm == "v0" else base)
            builds = {a: rung_sweep.build_json(d) for a, d in arms.items()}
            choice = rung_sweep.score(plan, runs, rung_sweep.build_json(ref), builds, vocab=256)
            by = {r["rows"]: r for r in choice["rungs"]}
            self.assertEqual((by[128]["winner"], by[1024]["winner"]), ("0", "1"))
            self.assertIn("FP32 gate", by[128]["status"])
            self.assertEqual(by[1024]["status"], "keep")
            self.assertTrue(by[128]["reference_program_matches"])
            self.assertFalse(choice["promotable"])
            lg = by[128]["arms"]["v0"]["logits"]
            self.assertTrue(lg["complete"] and lg["finite"] and not lg["bitwise"] and lg["top1"] == 1.0)
            recipe = {"emit": {"env": {"PLOW_NO_GLU_FUSE": "1"},
                               "rung_env": {"prefill": {"1024": {"PLOW_NO_GLU_FUSE": "0"}}}}}
            snip = rung_sweep.snippet(choice, recipe)
            self.assertIn('[emit.rung_env.prefill.128]\nPLOW_NO_GLU_FUSE = "0"', snip)
            self.assertIn("# delete PLOW_NO_GLU_FUSE from [emit.rung_env.prefill.1024]", snip)
            self.assertEqual(rung_sweep.switches(choice, recipe), {128: "0"})
            self.assertIn("| 128 |", rung_sweep.table(choice))
            # An arm without logits can never win; neither can one whose program equals the reference.
            for b in range(2):
                (runs / f"r128-b{b}-v0.logits" / "manifest.json").unlink()
            self.assertEqual({r["rows"]: r["winner"] for r in rung_sweep.score(
                plan, runs, rung_sweep.build_json(ref), builds, vocab=256)["rungs"]}[128], "1")
            same = {"v0": builds["v1"], "v1": builds["v1"]}
            r128 = rung_sweep.score(plan, runs, rung_sweep.build_json(ref), same, vocab=256)["rungs"][0]
            self.assertIn("no effect", r128["arms"]["v0"]["verdict"])
            # A reference arm whose program is not the reference build's decides nothing.
            other = {"v0": builds["v0"], "v1": builds["v0"]}
            r = rung_sweep.score(plan, runs, rung_sweep.build_json(ref), other, vocab=256)["rungs"][1]
            self.assertIsNone(r["winner"])
            self.assertIn("reference arm's program differs", r["status"])
            self.assertNotIn("delete", rung_sweep.snippet({**choice, "rungs": [by[128], r]}, recipe),
                             "an undecided rung keeps its override")
            # A reference that does not repeat across blocks decides nothing without thresholds.
            for f in (runs / "r1024-b1-v1.logits").iterdir():
                f.unlink()
            (runs / "r1024-b1-v1.logits").rmdir()
            (runs / "r1024-b1-v1.log").unlink()
            write_run(runs, "r1024-b1-v1", plan["rungs"][1], timings[(1024, "v1")], near)
            r = rung_sweep.score(plan, runs, rung_sweep.build_json(ref), builds, vocab=256)["rungs"][1]
            self.assertIsNone(r["winner"])
            self.assertIn("not repeatable", r["status"])
            r = rung_sweep.score(plan, runs, rung_sweep.build_json(ref), builds, 1.0, 0.9, vocab=256)["rungs"][1]
            self.assertEqual(r["winner"], "1")
            with self.assertRaises(ValueError):
                rung_sweep.score(plan, runs, rung_sweep.build_json(ref), builds, float("nan"), 0.9, vocab=256)

    def test_logit_captures_must_be_complete_and_equally_sized(self):
        with tempfile.TemporaryDirectory() as t:
            runs = Path(t)
            rung = {"rows": 128, "slots": 1, "ctx": 128}
            write_run(runs, "a", rung, [1.0, 1.0], [2.0, 1.0])
            write_run(runs, "b", rung, [1.0, 1.0], [2.0])
            cmp = rung_sweep.compare_logits
            self.assertFalse(cmp(runs / "a.logits", runs / "b.logits", 1, 2)["complete"])
            self.assertFalse(cmp(runs / "a.logits", runs / "a.logits", 2, 2)["complete"])
            self.assertFalse(cmp(runs / "a.logits", runs / "missing.logits", 1, 2)["complete"])
            self.assertFalse(cmp(runs / "b.logits", runs / "b.logits", 1, 2)["complete"], "short of the vocab")
            self.assertFalse(cmp(runs / "a.logits", runs / "a.logits", 1, None)["complete"])
            self.assertTrue(cmp(runs / "a.logits", runs / "a.logits", 1, 2)["bitwise"])
            (runs / "b.logits" / "manifest.json").write_text("[]")
            self.assertIn("malformed", cmp(runs / "a.logits", runs / "b.logits", 1, 2)["why"])
            w = rung_sweep.worst([{"complete": True, "finite": True, "bitwise": True, "cases": 1, "top1": 1.0,
                                   "kl_mean": 0.0, "kl_max": 0.0, "max_dlogit": 0.0}, {"complete": False, "why": "x"}])
            self.assertFalse(w["finite"])

    def test_qualification_needs_accuracy_resources_roofline_and_library_records(self):
        choice = {"rungs": [{"rows": 128, "winner": "0", "winner_arm": "v0", "reference_arm": "v1",
                             "accuracy_accepted": False, "status": "switch"}]}
        known = {"reg": 128, "stack": 0, "shared": 0}
        res = {"v0": {"a.cubin": {"sha256": "1", "local": 16, **known}, "b.cubin": {"sha256": "3", "error": "x"}},
               "v1": {"a.cubin": {"sha256": "2", "local": 0, **known}}}
        q = rung_sweep.qualify(choice, res, None, None)
        self.assertEqual(q["label"], "experimental")
        why = " ".join(q["why"])
        for part in ("accuracy not accepted", "spills", "resources unknown", "roofline", "library-kernel"):
            self.assertIn(part, why)
        with tempfile.NamedTemporaryFile() as f:
            ok = {"rungs": [dict(choice["rungs"][0], accuracy_accepted=True)]}
            clean = {"v0": {"a.cubin": {"sha256": "1", "local": 0, **known}}, "v1": res["v1"]}
            self.assertEqual(rung_sweep.qualify(ok, clean, f.name, f.name)["label"], "experimental", "empty record")
            f.write(b"{}"); f.flush()
            self.assertEqual(rung_sweep.qualify(ok, clean, f.name, f.name)["label"], "qualified")

    def test_noise_rule_is_section_seven(self):
        self.assertTrue(rung_sweep.noise_rule([17.2, 17.3], [15.0, 15.1])["faster"])
        self.assertFalse(rung_sweep.noise_rule([17.2, 17.2], [15.0, 17.0])["convictable"])
        self.assertFalse(rung_sweep.noise_rule([100.0, 100.0], [90.0, 90.5])["convictable"], "strict: zero drift")
        self.assertFalse(rung_sweep.noise_rule([17.2, 21.9, 17.3], [15.0, 15.1, 15.0])["faster"], "middle blocks count")
        self.assertFalse(rung_sweep.noise_rule([17.2, 17.3], [17.1, 17.2])["faster"])

    def test_resource_table_lists_only_differing_objects(self):
        res = {"v0": {"a.cubin": {"sha256": "1", "reg": 255, "stack": 960, "shared": 0, "local": 8}, "b.cubin": {"sha256": "2"}},
               "v1": {"a.cubin": {"sha256": "3", "reg": 128, "stack": 0, "shared": 0, "local": 0}, "b.cubin": {"sha256": "2"}}}
        t = rung_sweep.resource_table(res)
        self.assertIn("| a.cubin | REG 255 STACK 960 SHARED 0 LOCAL 8 SPILL | REG 128", t)
        self.assertNotIn("| b.cubin", t)
        self.assertIn("v0:a.cubin", t)


class BuildRungEnvTests(unittest.TestCase):
    def test_one_string_reaches_both_emits_and_other_paths_are_refused(self):
        r = {"emit": {"env": {}, "rung_env": {"prefill": {"128": {"PLOW_NO_GLU_FUSE": "1"}}}}}
        s = campaign.build_rung_env(r, {}, {}, ["prefill:128:PLOW_NO_GLU_FUSE=0", "prefill:4096:PLOW_NO_GLU_FUSE=0"])
        self.assertEqual(s, "prefill:128:PLOW_NO_GLU_FUSE=0;prefill:4096:PLOW_NO_GLU_FUSE=0")
        for recipe, environ, overrides, cli in [
            ({"emit": {"env": {"PLOW_EMIT_RUNG_ENV": "x"}}}, {}, {}, None),
            ({"emit": {}, "emit_roles": {"env": {"PLOW_EMIT_RUNG_ENV": "x"}}}, {}, {}, None),
            ({"emit": {}}, {"PLOW_EMIT_RUNG_ENV": "x"}, {}, None),
            ({"emit": {}}, {}, {"PLOW_EMIT_RUNG_ENV": "x"}, None),
            ({"emit": {"args": ["--emit-rung-env=x"]}}, {}, {}, None),
            ({"emit": {}}, {}, {}, ["prefill:128:PLOW_FP8_KV=1"]),
        ]:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                campaign.build_rung_env(recipe, environ, overrides, cli)

    def test_build_passes_and_records_the_rung_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / "target/release/plowc"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            recipe = root / "recipe.toml"
            recipe.write_text('[cell]\nhf_dir="weights"\ngpu="H100"\narch="sm_90a"\nn_cu=132\n'
                              '[emit]\n[emit.rung_env.prefill.128]\nPLOW_NO_GLU_FUSE="1"\n'
                              '[emit_roles.env]\nPLOW_X="1"\n[bench]\n')
            seen = []

            def run(command, env, log):
                seen.append(env.get("PLOW_EMIT_RUNG_ENV"))
                if "--out" in command:
                    assets = Path(command[command.index("--out") + 1])
                    (assets / "model.pkt").write_bytes(b"pkt")
                    (assets / "plow_config.h").write_bytes(b"cfg")
                return 0

            args = argparse.Namespace(recipe=str(recipe), out=str(root / "build"), env=[], no_probe=True,
                                      rung_env=["prefill:4096:PLOW_NO_GLU_FUSE=0"])
            with patch.object(campaign, "REPO", root), patch.object(campaign, "run", run), \
                    patch.object(campaign, "git", return_value="c"), \
                    patch.object(campaign, "lean_qualify", return_value=True), \
                    patch.dict(campaign.os.environ, {"CARGO_TARGET_DIR": str(root / "target")}):
                campaign.cmd_build(args)
            want = "prefill:128:PLOW_NO_GLU_FUSE=1;prefill:4096:PLOW_NO_GLU_FUSE=0"
            self.assertEqual(seen, [want, want])
            record = json.loads((root / "build/build-record.json").read_text())
            self.assertEqual(record["compilation"]["rung_env"], want)
            self.assertEqual(record["rung_overrides"], ["prefill:4096:PLOW_NO_GLU_FUSE=0"])



class SweepReuseTests(unittest.TestCase):
    def test_a_finished_build_is_reused_only_while_tree_compiler_and_artifacts_match(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "assets").mkdir()
            (out / "assets" / "model.pkt").write_bytes(b"pkt")
            plowc = out / "release" / "plowc"
            plowc.parent.mkdir()
            plowc.write_bytes(b"plowc")
            rec = {"commit": "abc", "dirty": False, "compiler_sha256": campaign.sha(plowc),
                   "build_provenance": {}, "compilation": {"toolchain_env": {}},
                   "hashes": {"model.pkt": campaign.sha(out / "assets" / "model.pkt")}, "objects": {}}
            git = {"status": "", "rev-parse": "abc"}
            env = {"CARGO_TARGET_DIR": d}
            with patch.object(campaign, "git", lambda *a, **k: git[a[0]]), patch.dict("os.environ", env, clear=True):
                self.assertEqual(campaign.sweep_build_stale(rec, out), "")
                (out / "assets" / "model.pkt").write_bytes(b"other")
                self.assertIn("model.pkt changed", campaign.sweep_build_stale(rec, out))
                (out / "assets" / "model.pkt").write_bytes(b"pkt")
                git["rev-parse"] = "def"
                self.assertIn("built at abc", campaign.sweep_build_stale(rec, out))
                git["rev-parse"] = "abc"
                plowc.write_bytes(b"new")
                self.assertEqual(campaign.sweep_build_stale(rec, out), "plowc changed")
                plowc.write_bytes(b"plowc")
                git["status"] = " M x"
                self.assertEqual(campaign.sweep_build_stale(rec, out), "the source tree is dirty")

if __name__ == "__main__":
    unittest.main()
