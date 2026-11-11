import unittest
import tempfile
from pathlib import Path

import check_recipe as c
import recipe_common as rc


class MigrationTests(unittest.TestCase):
    def test_frozen_allowlist_is_missing_only(self):
        for rel, digest in c.LEGACY_SPEECH_POLICIES.items():
            path = c.REPO / rel
            recipe = rc.load_toml(path)
            if "speech_tags" in recipe:
                self.assertEqual(c.check_speech_tags(recipe, path), [])
                continue
            self.assertEqual(rc.sha256_file(path), digest)
            self.assertIn("missing policy", c.check_speech_tags(recipe, path)[0])
            with tempfile.TemporaryDirectory() as tmp:
                duplicate = Path(tmp) / path.name
                duplicate.write_bytes(path.read_bytes())
                with self.assertRaisesRegex(rc.Fail, "missing policy"):
                    c.check_speech_tags(recipe, duplicate)
            recipe["speech_tags"] = {"waiver": "TBD"}
            with self.assertRaises(rc.Fail):
                c.check_speech_tags(recipe, path)

    def test_existing_self_test(self):
        c.self_test()


if __name__ == "__main__":
    unittest.main()
