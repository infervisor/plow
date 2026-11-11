import unittest
import render_recipe as render
import recipe_common as rc


class RenderPolicyTests(unittest.TestCase):
    def test_map_render_and_staleness(self):
        r = dict(name="model", label="test", emit=dict(args=[]), speech_tags=dict(version=1, profile="strip-silence", map=rc.default_speech_tag_map()))
        out = render.render(render.scaffold(r), r)
        self.assertIn('"markers": "strip"', out)
        self.assertIn('"pause"', out)
        self.assertEqual(render.render(out, r), out)
        r["speech_tags"]["map"]["tags"]["sigh"] = dict(reject=True)
        self.assertNotEqual(render.render(out, r), out)

    def test_waiver_render(self):
        r = dict(name="model", label="test", emit=dict(args=[]), speech_tags=dict(version=1, waiver="CER 0.2 vs 0.01", evidence=["ab.json"]))
        self.assertIn("CER 0.2 vs 0.01", render.render(render.scaffold(r), r))

    def test_existing_self_test(self):
        render.self_test()


if __name__ == "__main__":
    unittest.main()
