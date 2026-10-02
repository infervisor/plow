import unittest
from unittest.mock import patch

import plow_isa


class MarkerTests(unittest.TestCase):
    def test_section_relative_cuda_and_host_markers(self):
        sections = {"1": ".data", "2": ".nv.global.init", "3": ".nv.global", "4": ".bss"}
        symbols = [(0, 4, "OBJECT", str(i), name) for i, name in
                   [(1, "host"), (2, "cuda"), (3, "cuda_zero"), (4, "host_zero")]]
        def dump(command, context):
            return "0x00000000 20000000 " if "-x.nv.global.init" in command else "0x00000000 10000000 "
        with patch.object(plow_isa, "_sections", return_value=sections), \
             patch.object(plow_isa, "_symtab", return_value=symbols), \
             patch.object(plow_isa, "tool", return_value="readelf"), \
             patch.object(plow_isa, "_run", side_effect=dump):
            self.assertEqual(plow_isa.globals_u32("object"),
                             dict(host=16, cuda=32, cuda_zero=0, host_zero=0))
