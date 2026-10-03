import argparse
import json
from pathlib import Path
import tempfile
import unittest

import campaign


class GatePacketTests(unittest.TestCase):
    def fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        (assets / "model.pkt").write_bytes(b"packet")
        runtime = root / "plowrt"
        runtime.write_bytes(b"runtime")
        recipe = root / "recipe.toml"
        recipe.write_text('[gates.asr_wer]\nmanifest = "{repo}/m.jsonl"\nwer_max = 0.1\n')
        return argparse.Namespace(recipe=str(recipe), assets=str(assets), out=str(root / "gate"), only=None,
                                  plowrt=str(runtime), env=None, label=None, timeout=None, dry_run=False,
                                  score_only=False)

    def capture(self, a):
        a.dry_run = True
        campaign.cmd_gate(a)
        a.dry_run, a.score_only = False, True
        d = Path(a.out) / "asr_wer"
        (d / "served.jsonl").write_text(json.dumps({"wer": 0.01, "errors": 0}) + "\n")

    def test_score_only_uses_captured_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            campaign.cmd_gate(a)
            rec = json.loads((Path(a.out) / "gates.json").read_text())
            self.assertEqual(rec["packet_sha256"], campaign.sha(Path(a.assets) / "model.pkt"))
            self.assertTrue(rec["pass"])

    def test_score_only_refuses_other_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            (Path(a.assets) / "model.pkt").write_bytes(b"rebuilt packet")
            with self.assertRaises(SystemExit):
                campaign.cmd_gate(a)
            self.assertFalse((Path(a.out) / "gates.json").exists())

    def test_score_only_refuses_unrecorded_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fixture(root)
            self.capture(a)
            (Path(a.out) / "packet.sha256").unlink()
            with self.assertRaises(SystemExit):
                campaign.cmd_gate(a)

    def test_lenient_expand_keeps_unset_env(self):
        out = Path("/x")
        self.assertEqual(campaign.expand("{env:GATE_TEST_UNSET}/{out}", out, lenient=True), "{env:GATE_TEST_UNSET}//x")
        with self.assertRaises(SystemExit):
            campaign.expand("{env:GATE_TEST_UNSET}", out)


if __name__ == "__main__":
    unittest.main()
