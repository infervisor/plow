import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class AmdTokenBatchBuildTests(unittest.TestCase):
    def configure(self, directory, arch, gq):
        directory = pathlib.Path(directory)
        config = directory / "packet.h"
        config.write_text("#define PLOW_PACKET_HASH_LO 123\n")
        subprocess.run(
            [
                "cmake", "-S", str(ROOT / "runtime"), "-B", str(directory / "build"),
                "-G", "Unix Makefiles", "-DPLOW_GFX950_HSACO=ON",
                f"-DPLOW_HSACO_ARCH={arch}", f"-DPLOW_HSACO_GQ={gq}",
                f"-DPLOW_HSACO_CONFIG={config}",
                "-DPLOW_HSACO_EXTRA_DEFINES=-DPLOW_TEST_PACKET_EXTRA=1",
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        build = (directory / "build/CMakeFiles/gfx950_hsaco.dir/build.make").read_text()
        commands = {}
        for line in build.splitlines():
            if "hipcc_hsaco.sh" in line and line.startswith("\t"):
                words = shlex.split(line)
                wrapper = next(i for i, word in enumerate(words) if word.endswith("hipcc_hsaco.sh"))
                args = words[wrapper + 1:]
                commands[pathlib.Path(args[3]).name] = args
        return build, commands

    def test_gfx942_default_build_contains_universal_token_batch_twins(self):
        with tempfile.TemporaryDirectory() as directory:
            build, commands = self.configure(directory, "gfx942", "ON")
            for suffix in ("", "_gq"):
                name = f"interp_tokbatch{suffix}.elf"
                self.assertIn(name, set(commands))
                args = commands[name]
                self.assertEqual(args[4:7], [f"plow_interp_tokbatch_gfx942{suffix}", "512", "1"])
                for define in (
                    "-DPLOW_TOKEN_BATCH=1", "-DPLOW_MIXED_STEP=1", "-DPLOW_WG_WAVES=4",
                    "-DGM_BM=256", "-DGM_BN=128",
                ):
                    self.assertIn(define, args)
                self.assertEqual("-DPLOW_GLOBAL_QUEUE=1" in args, suffix == "_gq")
                self.assertFalse(any("PLOW_CONFIG=" in arg for arg in args))
                self.assertNotIn("-DPLOW_TEST_PACKET_EXTRA=1", args)
                self.assertIn(f"gfx950_hsaco: hsaco/{name}", build)
            ordinary = commands["interp_prefill.elf"]
            self.assertTrue(any("PLOW_CONFIG=" in arg for arg in ordinary))
            self.assertIn("-DPLOW_TEST_PACKET_EXTRA=1", ordinary)

    def test_gq_disable_and_arch_selection_preserve_supported_outputs(self):
        for arch, gq, expected in (("gfx942", "OFF", {"interp_tokbatch.elf"}), ("gfx950", "ON", set())):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                _, commands = self.configure(directory, arch, gq)
                self.assertEqual({name for name in commands if name.startswith("interp_tokbatch")}, expected)


if __name__ == "__main__":
    unittest.main()
