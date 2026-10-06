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
    def test_every_recipe_serve_env_yields_loadable_serve_defaults(self):
        import tomllib
        known = campaign.runtime_knobs()
        self.assertIn("PLOW_PF_INTERLEAVE", known)
        recipes = sorted((campaign.REPO / "recipes").rglob("*.toml")) + sorted(
            (campaign.REPO / "scripts/campaign/recipes").glob("*.toml"))
        self.assertTrue(recipes)
        for path in recipes:
            r = tomllib.loads(path.read_text())
            keep, _ = campaign.packet_serve_defaults(r, Path("/tmp/out"))
            for k, v in keep.items():
                self.assertTrue(k.startswith("PLOW_") and k in known, f"{path}: {k}")
                self.assertNotIn("/", v, f"{path}: {k}")
                self.assertNotIn(",", v, f"{path}: {k}")
        keep, skipped = campaign.packet_serve_defaults(
            {"serve": {"env": {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "PLOW_LIBCUDA": "/usr/lib/libcuda.so",
                               "PLOW_PF_INTERLEAVE": "2048", "PLOW_PREFIX_CACHE": "1"}}}, Path("/tmp/out"))
        self.assertEqual(keep, {"PLOW_PF_INTERLEAVE": "2048"})
        self.assertEqual(sorted(skipped), ["LD_LIBRARY_PATH", "PLOW_LIBCUDA"])

    def test_role_rebuild_preserves_overrides_and_copies_final_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / "target/release/plowc"
            compiler.parent.mkdir(parents=True)
            compiler.write_bytes(b"compiler")
            original_compiler_sha = campaign.sha(compiler)
            recipe = root / "recipe.toml"
            recipe.write_text('[cell]\nhf_dir="weights"\ngpu="H100"\narch="sm_90a"\nn_cu=132\n'
                              '[emit]\n[emit_roles.env]\nPLOW_GEMMA4_SM90_W8A8_GEMM_GLU_ROLE="1"\n'
                              '[objects]\nscript="build.sh"\nrole_files=["role.cubin"]\n'
                              '[objects.env]\nPLOW_BUILD_W8A8="0"\nNVCC_APPEND_FLAGS="-DPLOW_NV_GLU_QUANT_CACHE=1"\n'
                              '[bench]\n')
            original_recipe = recipe.read_bytes()
            (root / "flake.lock").write_bytes(b"original lock")
            out = root / "build"
            configs = []

            def run(command, env, log):
                recipe.write_bytes(b"recipe changed during compilation")
                compiler.write_bytes(b"compiler replaced during compilation")
                (root / "flake.lock").write_bytes(b"changed lock")
                with log.open("a") as f:
                    f.write(json.dumps(command) + "\n")
                if "--out" in command:
                    assets = Path(command[command.index("--out") + 1])
                    (assets / "model.pkt").write_bytes(assets.name.encode())
                    (assets / "plow_config.h").write_bytes(assets.name.encode())
                    if assets.name == "assets":
                        self.assertEqual((assets / "role.cubin").read_bytes(), b"base")
                        (assets / "role.cubin").write_bytes(b"emit placeholder")
                else:
                    self.assertEqual(command[:2], ["bash", "-x"])
                    self.assertEqual(env["PLOW_BUILD_W8A8"], "1")
                    self.assertEqual(env["NVCC_APPEND_FLAGS"], "-DPLOW_NV_GLU_QUANT_CACHE=1 -ccbin=/usr/bin/g++-14")
                    config = Path(env["PLOW_CUBIN_CONFIG"]).read_bytes()
                    configs.append(config)
                    objects = Path(command[-1])
                    objects.mkdir()
                    (objects / "role.cubin").write_bytes(config)
                return 0

            args = argparse.Namespace(recipe=str(recipe), out=str(out), env=[], no_probe=True,
                                      object_env=["PLOW_BUILD_W8A8=1", "NVCC_APPEND_FLAGS=-ccbin=/usr/bin/g++-14"])
            with patch.object(campaign, "REPO", root), patch.object(campaign, "run", run), \
                    patch.object(campaign, "git", side_effect=lambda *args, **kwargs:
                                 "start" if recipe.read_bytes() == original_recipe else "end"), \
                    patch.dict(campaign.os.environ, {"CARGO_TARGET_DIR": str(root / "target")}):
                campaign.cmd_build(args)
            self.assertEqual(configs, [b"base", b"assets"])
            self.assertEqual((out / "objects-base/role.cubin").read_bytes(), b"base")
            self.assertEqual((out / "assets/role.cubin").read_bytes(), b"assets")
            record = json.loads((out / "build-record.json").read_text())
            self.assertEqual(record["commit"], "start")
            self.assertEqual(record["compiler_sha256"], original_compiler_sha)
            provenance = record["build_provenance"]
            self.assertEqual(provenance["source_end"]["commit"], "end")
            self.assertTrue(provenance["source_state_changed"])
            self.assertTrue(provenance["compiler_changed"])
            self.assertEqual(provenance["compiler_end_sha256"], campaign.sha(compiler))
            self.assertEqual((out / "source-start.diff").read_text(), "start")
            self.assertEqual((out / "source-end.diff").read_text(), "end")
            self.assertEqual((out / "recipe.toml").read_bytes(), original_recipe)
            self.assertEqual(record["recipe_sha256"], campaign.sha(out / "recipe.toml"))
            self.assertNotEqual(record["recipe_sha256"], campaign.sha(recipe))
            self.assertEqual((out / "flake.lock").read_bytes(), b"original lock")
            self.assertEqual(record["compilation"]["flake_lock_sha256"], campaign.sha(out / "flake.lock"))
            self.assertEqual(record["hashes"]["role.cubin"], record["objects"]["role.cubin"])
            self.assertEqual(record["compilation"]["log_sha256"], campaign.sha(out / "build.log"))
            self.assertEqual(record["compilation"]["object_env"]["PLOW_BUILD_W8A8"], "1")
            self.assertEqual(record["compilation"]["object_env"]["NVCC_APPEND_FLAGS"],
                             "-DPLOW_NV_GLU_QUANT_CACHE=1 -ccbin=/usr/bin/g++-14")

    def test_object_env_appends_flags_and_replaces_scalars(self):
        recipe = {"NVCC_APPEND_FLAGS": "-DPLOW_NV_GLU_QUANT_CACHE=1 -DPLOW_NV_GLU_QUANT_WPR=1",
                  "PLOW_BUILD_W8A8": "1", "PLOW_BUILD_FATLITE": 1}
        merged = campaign.merge_object_env(recipe, {"NVCC_APPEND_FLAGS": "-ccbin=/usr/bin/g++-14",
                                                    "PLOW_BUILD_W8A8": "0", "CFLAGS": "-O2"})
        self.assertEqual(merged, {
            "NVCC_APPEND_FLAGS": "-DPLOW_NV_GLU_QUANT_CACHE=1 -DPLOW_NV_GLU_QUANT_WPR=1 -ccbin=/usr/bin/g++-14",
            "PLOW_BUILD_W8A8": "0", "PLOW_BUILD_FATLITE": "1", "CFLAGS": "-O2"})
        self.assertEqual(campaign.merge_object_env(recipe, {}), {k: str(v) for k, v in recipe.items()})

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
