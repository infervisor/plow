import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CMAKE = (ROOT / "runtime" / "CMakeLists.txt").read_text()


def cmake_set(name):
    match = re.search(rf"set\({name}\s+([^)]*)\)", CMAKE)
    if match is None:
        raise AssertionError(f"missing CMake set({name} ...)")
    return match.group(1).split()


class NvidiaCubinAxisTests(unittest.TestCase):
    def test_packet_config_receives_the_object_phase(self):
        self.assertIn("-DPLOW_BUCKET_DECODE=1", cmake_set("_ax_decode"))
        for axis in ("_ax_prefill", "_ax_seg_fat", "_ax_seg_gemm"):
            self.assertIn("-DPLOW_BUCKET_DECODE=0", cmake_set(axis))


if __name__ == "__main__":
    unittest.main()
