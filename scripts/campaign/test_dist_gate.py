import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import campaign

# push copies the assets (and objects/) into the file:// registry; load copies the registry into
# $PLOW_HOME/m; ls --json names that dir; inventory lists a dir's file names.
FAKE_PLOWRT = r"""#!/usr/bin/env bash
set -e
case "$1" in
push) shift; while [ $# -gt 0 ]; do case "$1" in --assets) A=$2;; --objects) O=$2;; --to) T=${2#file://};; esac; shift 2; done
      mkdir -p "$T/objects"; cp "$A"/* "$T"/; [ "$O" = "$A" ] || cp "$O"/* "$T/objects/";;
load) mkdir -p "$PLOW_HOME"; cp -r "${PLOW_REGISTRY#file://}" "$PLOW_HOME/m"; echo "$@" > "$PLOW_HOME/load.args";;
ls) echo "[{\"name\": \"model-a\", \"dir\": \"$PLOW_HOME/m\"}]";;
inventory) (cd "$2" && ls *.pkt);;
esac
"""


class DistGateTests(unittest.TestCase):
    def fixture(self, root):
        build = root / "build"
        (build / "assets").mkdir(parents=True)
        (build / "objects").mkdir()
        (build / "assets" / "model.pkt").write_bytes(b"packet")
        (build / "assets" / "build.json").write_text("{}")
        (build / "objects" / "k.cubin").write_bytes(b"obj")
        rec = root / "recipes" / "model-a" / "h100.toml"
        rec.parent.mkdir(parents=True)
        rec.write_text('[cell]\nn_gpu = 1\n[checkpoint]\nsource = "local"\n')
        bin_ = root / "plowrt"
        bin_.write_text(FAKE_PLOWRT)
        bin_.chmod(0o755)
        return argparse.Namespace(recipe=str(rec), assets=str(build / "assets"), out=str(root / "out"), objects=None,
                                  plowrt=str(bin_), checkpoint=str(root / "ckpt"), name=None, gate=False, only=None,
                                  queue=str(root / "q"))

    def smoke(self, served: str, rc: int):
        def fake(label, ngpu, cmd, log, queue):
            script = Path(cmd[-1]).read_text()
            self.assertIn("serve --model model-a", script)
            self.assertIn("export PLOW_HOME=", script)
            self.assertIn("export PLOW_REGISTRY=file://", script)
            Path(log).write_text("SMOKE\n")
            (Path(cmd[-1]).parent / "serve.log").write_text(served)
            return rc
        return mock.patch.object(campaign, "queued", side_effect=fake)

    def test_pass_records_tree_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            with self.smoke("ready\n", 0):
                campaign.cmd_dist_gate(a)
            rec = json.loads((Path(a.out) / "dist-gate.json").read_text())
            self.assertTrue(rec["pass"])
            self.assertEqual(rec["tree"], {"files": 3, "missing": [], "differ": []})
            self.assertTrue(rec["inventory_equal"])
            self.assertIn("--checkpoint", (Path(a.out) / "home" / "load.args").read_text())

    def test_speech_legacy_admission_and_policy_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            path = root / "recipes/infervisor/maya2-native/sm90a-h100-tp1.toml"
            path.parent.mkdir(parents=True)
            source = Path(a.recipe).read_text() + '\n[bench]\nkind = "speech"\n'
            path.write_text(source)
            a.recipe = str(path)
            a.name = "model-a"
            for edit in ("", '\n# routine re-pin\n'):
                path.write_text(source + edit)
                stderr = io.StringIO()
                with self.smoke("ready\n", 0), contextlib.redirect_stderr(stderr):
                    campaign.cmd_dist_gate(a)
                self.assertIn("speech_tags: legacy recipe without tag policy (grandfathered 2026-11-11)", stderr.getvalue())
                self.assertTrue(json.loads((Path(a.out) / "dist-gate.json").read_text())["pass"])
            unlisted = path.with_name("new.toml")
            unlisted.write_text(source)
            a.recipe = str(unlisted)
            with mock.patch.object(campaign, "run") as run, self.assertRaisesRegex(campaign.recipe_common.Fail, "missing policy"):
                campaign.cmd_dist_gate(a)
            run.assert_not_called()
            a.recipe = str(path)
            path.write_text(source + '\n[speech_tags]\nversion = 1\nwaiver = "TBD"\n')
            with mock.patch.object(campaign, "run") as run, self.assertRaises(campaign.recipe_common.Fail):
                campaign.cmd_dist_gate(a)
            run.assert_not_called()

    def test_differing_object_fails_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            real = campaign.run
            def corrupt(cmd, env, log):
                rc = real(cmd, env, log)
                if cmd[1] == "load":
                    (Path(env["PLOW_HOME"]) / "m" / "objects" / "k.cubin").write_bytes(b"other")
                return rc
            with mock.patch.object(campaign, "run", side_effect=corrupt), \
                 mock.patch.object(campaign, "queued") as q, self.assertRaises(SystemExit):
                campaign.cmd_dist_gate(a)
            q.assert_not_called()
            rec = json.loads((Path(a.out) / "dist-gate.json").read_text())
            self.assertEqual(rec["tree"]["differ"], ["objects/k.cubin"])
            self.assertFalse(rec["pass"])

    def test_serve_fault_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.fixture(Path(tmp))
            with self.smoke("thread panicked at x\n", 0), self.assertRaises(SystemExit):
                campaign.cmd_dist_gate(a)
            self.assertEqual(json.loads((Path(a.out) / "dist-gate.json").read_text())["faults"], 1)


class QueuedTests(unittest.TestCase):
    def test_waits_for_job_and_appends_its_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpuq = root / "gpuq.py"
            # Submits a job that is already finished, as the runner would leave it.
            gpuq.write_text("import json, sys\nfrom pathlib import Path\n"
                            "r = Path(sys.argv[2]); r.mkdir(exist_ok=True)\n"
                            "(r / 'j1.json').write_text(json.dumps({'state': 'failed', 'rc': 7, 'command': sys.argv[6:]}))\n"
                            "(r / 'j1.log').write_text('job output\\n'); print('j1')\n")
            log = root / "run.log"
            with mock.patch.object(campaign, "GPUQ", gpuq):
                rc = campaign.queued("lbl", 1, ["bash", "x.sh"], log, str(root / "q"))
            self.assertEqual(rc, 7)
            self.assertIn("job output", log.read_text())
            self.assertEqual(json.loads((root / "q" / "j1.json").read_text())["command"], ["bash", "x.sh"])


if __name__ == "__main__":
    unittest.main()
