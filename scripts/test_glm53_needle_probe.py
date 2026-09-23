import unittest

from glm53_needle_probe import exact_prompt


class ExactNeedleTests(unittest.TestCase):
    def test_exact_lengths_keep_needle_and_question(self):
        tokenize = lambda text: list(text.encode())
        for length in (8192, 71680):
            for depth in (0.1, 0.5, 0.9):
                ids = exact_prompt(tokenize, length, depth, "UNIQUE NEEDLE", "QUESTION?")
                self.assertEqual(len(ids), length)
                self.assertEqual(bytes(ids).count(b"UNIQUE NEEDLE"), 1)
                self.assertTrue(bytes(ids).endswith(b"QUESTION?"))
                self.assertEqual(ids, exact_prompt(tokenize, length, depth, "UNIQUE NEEDLE", "QUESTION?"))

    def test_invalid_geometry_refused(self):
        for length, depth in ((1, 0.5), (100, -0.1), (100, 1.1)):
            with self.assertRaises(ValueError):
                exact_prompt(lambda text: list(text.encode()), length, depth, "needle", "question")


if __name__ == "__main__":
    unittest.main()
