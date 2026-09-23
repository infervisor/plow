import argparse
import json
from pathlib import Path
import tempfile
import struct
import subprocess
import contextlib
import io
import unittest
from unittest.mock import patch

import campaign


class CampaignBuildTests(unittest.TestCase):
    def test_block_roofline_trace_binds_packet_runtime_and_counter_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, runtime = root / "model.pkt", root / "plowrt"
            packet.write_bytes(b"packet")
            runtime.write_bytes(b"runtime")
            recipe = root / "recipe.toml"
            recipe.write_text('[roofline]\nbandwidth_gbps=6200\nbf16_tflops=2300\n')
            trace = root / "trace.bin"
            trace.write_bytes(struct.pack("<IIIHHQQQ",0,0,0,9,0,1,1,11))
            source = root / "run-record.json"
            source.write_text(json.dumps({"packet_sha256":campaign.sha(packet),
                                          "runtime_sha256":campaign.sha(runtime)}))
            args = argparse.Namespace(recipe=str(recipe), packet=str(packet), plowrt=str(runtime),
                program=1, ctx=512, router_table=None, out=str(root / "out"),
                trace=str(trace), trace_clock_hz=1e9, trace_run_record=str(source))
            text = "===== program T=1 1 insts\n#0 Gemv b=1 C<-x | M=1 N=128 K=256\n"
            program = {"n_inst":1,"n_counter":1,"insts":[
                {"idx":0,"op":9,"op_name":"Gemv","blocks":1}],
                "counters":{"per_counter":[{"id":0,"producer":0,"threshold":1,"consumers":[]}]}}
            def run(command, **kwargs):
                output = json.dumps({"programs":[program]}) if "--format" in command else text
                return subprocess.CompletedProcess(command,0,output,"")
            with patch.object(campaign.subprocess,"run",run), patch.object(campaign,"git",return_value="test"), \
                    contextlib.redirect_stdout(io.StringIO()):
                campaign.cmd_block_roofline(args)
                report = json.loads((root / "out/roofline.json").read_text())
                self.assertEqual(report["trace_priorities"]["counter_chain_ns"],10)
                self.assertEqual(report["trace_run_record_sha256"],campaign.sha(source))
                source.write_text(json.dumps({"packet_sha256":"a"*64,"runtime_sha256":campaign.sha(runtime)}))
                with self.assertRaises(SystemExit):
                    campaign.cmd_block_roofline(args)

    def test_amd_build_pairs_config_and_records_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / "target/release/plowc"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            recipe = root / "recipe.toml"
            recipe.write_text('[cell]\nhf_dir="weights"\ngpu="MI350X"\narch="gfx950"\nn_cu=256\n'
                              '[emit]\nemit="devblob"\n[objects]\nscript="build.sh"\n[bench]\n')
            out = root / "build"
            calls = []

            def run(command, env, log):
                calls.append(command)
                if "--out" in command:
                    self.assertNotIn("PLOW_GEMV_MFMA4", env)
                    assets = Path(command[command.index("--out") + 1])
                    assets.mkdir()
                    for name in ("model.pkt", "build.json", "plow_config.h"):
                        (assets / name).write_bytes(name.encode())
                else:
                    self.assertEqual(env["PLOW_HSACO_CONFIG"], str(out / "assets/plow_config.h"))
                    self.assertEqual(env["PLOW_GEMV_MFMA4"], "1")
                    objects = Path(command[-1])
                    objects.mkdir()
                    (objects / "interp_decode.elf").write_bytes(b"object")
                return 0

            args = argparse.Namespace(recipe=str(recipe), out=str(out), env=[], no_probe=True,
                                      object_env=["PLOW_GEMV_MFMA4=1"])
            with patch.object(campaign, "REPO", root), patch.object(campaign, "run", run), \
                    patch.object(campaign, "git", return_value="test"):
                campaign.cmd_build(args)
            self.assertEqual(len(calls), 2)
            record = json.loads((out / "build-record.json").read_text())
            self.assertEqual(record["object_overrides"], {"PLOW_GEMV_MFMA4": "1"})
            self.assertEqual(record["objects"]["interp_decode.elf"], campaign.sha(out / "objects/interp_decode.elf"))
            self.assertEqual(record["hashes"]["build.json"], campaign.sha(out / "assets/build.json"))

    def test_block_freeze_rejects_changed_campaign_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "build/assets"
            assets.mkdir(parents=True)
            packet = assets / "model.pkt"
            packet.write_bytes(b"changed packet")
            (assets.parent / "build-record.json").write_text('{"hashes":{"model.pkt":"old hash"}}')
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "reference.bf16").write_bytes(b"reference")
            (inputs / "reference.json").write_text("{}")
            recipe = root / "recipe.toml"
            recipe.write_text('[cell]\nname="test"\n')
            args = argparse.Namespace(recipe=str(recipe), out=str(root / "run"), packet=str(packet),
                                      objects=str(root / "objects"), inputs=str(inputs), checkpoint="weights")
            with self.assertRaises(SystemExit):
                campaign.cmd_block_bench(args)

    def test_block_host_timing_is_explicit_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "block.pkt"
            packet.write_bytes(b"packet")
            runtime = root / "plowrt"
            runtime.write_bytes(b"runtime")
            objects = root / "objects"
            objects.mkdir()
            (objects / "decode.elf").write_bytes(b"object")
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "reference.bf16").write_bytes(b"reference")
            (inputs / "reference.json").write_text("{}")
            recipe = root / "recipe.toml"
            recipe.write_text('[cell]\nname="test"\nn_gpu=8\narch="gfx950"\n')
            args = argparse.Namespace(recipe=str(recipe), out=str(root / "run"), packet=str(packet),
                                      objects=str(objects), inputs=str(inputs), checkpoint="weights",
                                      plowrt=str(runtime), ctx=512, repeat=64, warmup=5,
                                      trace=False, dstep_log=True)
            with patch.object(campaign, "run", return_value=0), \
                    patch.object(campaign, "git", return_value="test"):
                record = campaign.prepare_block_bench(args)
            self.assertEqual(record["env"]["PLOW_DSTEP_LOG"], "1")
            self.assertEqual(record["env"]["PLOW_DSTEP_EVERY"], "32")
            self.assertIn("export PLOW_DSTEP_LOG=1", (root / "run/run.sh").read_text())


if __name__ == "__main__":
    unittest.main()
