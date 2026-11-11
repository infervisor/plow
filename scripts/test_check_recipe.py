import unittest
import tempfile
from pathlib import Path

import check_recipe as c
import recipe_common as rc


class MigrationTests(unittest.TestCase):
    def test_identity_allowlist_is_missing_only(self):
        inventory = rc.load_json(c.REPO / "crates/plow-asset/src/speech_tag_legacy.json")
        for rel, date in inventory.items():
            path = c.REPO / ("recipes/" + rel if rel.startswith("infervisor/") else rel)
            recipe = rc.load_toml(path)
            if "speech_tags" in recipe:
                self.assertEqual(c.check_speech_tags(recipe, path), [])
                continue
            notice = f"info: speech_tags: legacy recipe without tag policy (grandfathered {date})"
            self.assertEqual(c.check_speech_tags(recipe, path), [notice])
            recipe["plow_git"] = "a" * 40
            recipe.setdefault("gates", {})["tts_cer"] = {"cer_median_max": .07}
            self.assertEqual(c.check_speech_tags(recipe, path), [notice])
            with tempfile.TemporaryDirectory() as tmp:
                duplicate = Path(tmp) / path.name
                duplicate.write_bytes(path.read_bytes())
                with self.assertRaisesRegex(rc.Fail, "missing policy"):
                    c.check_speech_tags(recipe, duplicate)
            recipe["speech_tags"] = {"waiver": "TBD"}
            with self.assertRaises(rc.Fail):
                c.check_speech_tags(recipe, path)
            recipe["speech_tags"] = dict(version=1, profile="test", map=rc.default_speech_tag_map())
            self.assertEqual(c.check_speech_tags(recipe, path), [])

    def test_edited_legacy_file_is_still_checked_successfully(self):
        source = c.REPO / "recipes/infervisor/maya2-native/sm90a-h100-tp1.toml"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes/infervisor/maya2-native/sm90a-h100-tp1.toml"
            path.parent.mkdir(parents=True)
            recipe = rc.load_toml(source)
            recipe["plow_git"] = "a" * 40
            recipe["gates"]["tts_cer"]["cer_median_max"] = .07
            path.write_text(c.to_toml(recipe))
            notes = c.check(path, None, None, False)
            self.assertEqual(sum("speech_tags: legacy recipe" in n for n in notes), 1)
            recipe["speech_tags"] = dict(version=1, profile="test", map=rc.default_speech_tag_map())
            path.write_text(c.to_toml(recipe))
            self.assertFalse(any("speech_tags: legacy recipe" in n for n in c.check(path, None, None, False)))
            recipe["speech_tags"]["map"]["tags"].pop("sigh")
            path.write_text(c.to_toml(recipe))
            with self.assertRaises(rc.Fail):
                c.check(path, None, None, False)

    def test_existing_self_test(self):
        c.self_test()


if __name__ == "__main__":
    unittest.main()
