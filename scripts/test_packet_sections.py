import json
import struct
import tempfile
import unittest
from pathlib import Path

from packet_sections import read_section


def write_packet(path, pipelines):
    raw = json.dumps(dict(version=1, pipelines=pipelines)).encode()
    header = bytearray(64)
    header[:8] = b"PLOWDEV\x08"
    struct.pack_into("<Q", header, 40, 64 + len(raw))
    directory = b"SECT" + struct.pack("<I", 1) + struct.pack("<IIQQ24s", 4, 0, 64, len(raw), b"packet_pipeline.json")
    path.write_bytes(header + raw + directory)


class SectionTests(unittest.TestCase):
    def test_section_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pkt"
            write_packet(p, [])
            self.assertEqual(json.loads(read_section(p, "packet_pipeline.json"))["pipelines"], [])
            self.assertIsNone(read_section(p, "absent"))
            data = bytearray(p.read_bytes())
            data[-24:] = b"packet_pipeline.json\0\0\0\0\0"
            struct.pack_into("<Q", data, len(data) - 40, 2**63)
            p.write_bytes(data)
            with self.assertRaises(ValueError):
                read_section(p, "packet_pipeline.json")
            p.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                read_section(p, "packet_pipeline.json")


if __name__ == "__main__":
    unittest.main()
