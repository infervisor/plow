import unittest
from tag_discovery import aliases, discover, text_rules


class DiscoveryTests(unittest.TestCase):
    def tokenizer(self):
        return dict(model=dict(vocab={}), added_tokens=[dict(content="[sigh]", id=611, special=True),
                                                      dict(content="[hi]", id=722, special=False)])

    def pipeline(self):
        return dict(name="speech", driver="tts.guided_lm.v1", strings={
            "text.rules": "lowercase\nnfkd\nlanguage\nprefix\t[{lang}]",
            "text.rules.lang.en": "", "text.rules.lang.hi": ""})

    def test_events_language_and_nfkd_fullwidth(self):
        def encode(text):
            return [i for t, i in (("[sigh]", 611), ("[hi]", 722)) for _ in range(text.count(t))]
        report = discover(self.tokenizer(), [self.pipeline()], encode)
        collisions = report["pipelines"][0]["collisions"]
        for token, kind in (("[sigh]", "event"), ("[hi]", "language")):
            hits = [h for h in collisions if h["token"] == token]
            self.assertTrue(any(h["injected_count"] == 1 and h["form"] == "token" for h in hits))
            self.assertTrue(any(h["injected_count"] == 1 and h["compatibility_bypasses_parser"] for h in hits))
            self.assertTrue(all(h["kind"] == kind for h in hits))
        hi = next(h for h in collisions if h["input"] == "[hi]" and h["language"] == "hi")
        self.assertEqual(hi["baseline_ids"], [722])
        self.assertEqual(hi["ids"], [722, 722])
        self.assertIn(["[", "("], report["literal_rewrites_needed"])

    def test_unsupported_rules_are_incomplete(self):
        p = self.pipeline()
        p["strings"]["text.rules"] = "segment_crf\tzh.seg"
        report = discover(self.tokenizer(), [p])
        self.assertFalse(report["complete"])
        self.assertEqual(report["unsupported_rules"], ["segment_crf\tzh.seg"])
        self.assertTrue(all(h["ids"] is None for h in report["pipelines"][0]["collisions"]))

    def test_actual_alias_table_and_rules(self):
        self.assertEqual(aliases()["laughs softly"], "laugh")
        self.assertEqual(aliases()["cheerful"], "happy")
        text, unsupported = text_rules(" ［SIGH］  ", {"text.rules": "strip\nlowercase\nnfkd\nreplace\tsigh\tcalm"}, "hi")
        self.assertEqual(text, "[calm]")
        self.assertFalse(unsupported)


if __name__ == "__main__":
    unittest.main()
