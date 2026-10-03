import unittest
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

from glm53_needle_probe import NEEDLES, compare_captures, exact_prompt, main


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

    def test_exact_chat_length_includes_both_template_boundaries(self):
        ids = exact_prompt(lambda text: list(text.encode()), 512, 0.5,
                           "UNIQUE NEEDLE", "QUESTION?", "<bos>USER\n", "\nMODEL\n")
        self.assertEqual(len(ids), 512)
        self.assertTrue(bytes(ids).startswith(b"<bos>USER\n"))
        self.assertTrue(bytes(ids).endswith(b"QUESTION?\nMODEL\n"))
        self.assertEqual(bytes(ids).count(b"UNIQUE NEEDLE"), 1)

    def test_concurrent_capture_preserves_pairing_and_request_settings(self):
        barrier = threading.Barrier(2, timeout=5)
        requests = []

        def post(url, body):
            if url.endswith('/tokenize'):
                return {'tokens': list(body['prompt'].encode())}
            requests.append(body)
            barrier.wait()
            return {'usage': {'prompt_tokens': len(body['prompt'])},
                    'choices': [{'text': '7429 Trondheim 312'}]}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'capture.json'
            argv = ['probe', '--url', 'http://fake', '--arm', 'test', '--out', str(out),
                    '--lens', '512', '--depths', '0.5', '--exact-lengths',
                    '--concurrency', '2', '--repeats', '2', '--max-tokens', '128', '--ignore-eos']
            with patch('sys.argv', argv), patch('glm53_needle_probe.post', post), \
                 patch('urllib.request.urlopen', return_value=io.BytesIO(b'{"data":[{"id":"model"}]}')), \
                 patch('builtins.print'):
                main()
            result = json.loads(out.read_text())
        cells = result['cells']
        self.assertEqual(len(cells), 6)
        self.assertEqual(len({(c['item'], c['tokens'], c['repeat']) for c in cells}), 6)
        for first, second in zip(cells[::2], cells[1::2]):
            self.assertEqual(first['prompt_sha256_u32le'], second['prompt_sha256_u32le'])
            self.assertEqual((first['repeat'], second['repeat']), (0, 1))
        self.assertTrue(all(c['correct'] and c['prompt_tokens'] == 512 for c in cells))
        self.assertEqual(result['concurrency'], 2)
        self.assertTrue(all(r['ignore_eos'] and r['max_tokens'] == 128 and r['temperature'] == 0
                            for r in requests))


class PairedRetrievalTests(unittest.TestCase):
    def capture(self):
        return dict(concurrency=2, repeats=2, max_tokens=128, ignore_eos=True,
                    lengths=[512], depths=[0.5], cells=[
                        dict(tokens=512, prompt_tokens=512, depth=0.5, item=f'{nid}@0.5',
                             repeat=repeat, prompt_sha256_u32le=nid, expect=expect, text=expect)
                        for nid, _, _, expect in NEEDLES for repeat in range(2)])

    def test_scores_each_repeat_and_recomputes_correctness(self):
        ref = self.capture()
        cand = copy.deepcopy(ref)
        cand['cells'][1].update(text='wrong answer', correct=True)
        result = compare_captures(ref, cand)
        self.assertEqual(result['verdict'], 'fail')
        self.assertEqual(result['candidate_correct'], 5)
        self.assertEqual(result['regressions'], [(512, 0.5, 'code@0.5', 1)])
        self.assertEqual(result['unique_prompts'], 3)

    def test_rejects_incomplete_duplicate_mismatched_and_empty_captures(self):
        ref = self.capture()
        for mutate in (lambda c: c['cells'].pop(),
                       lambda c: c['cells'].append(c['cells'][0]),
                       lambda c: c['cells'][0].update(prompt_sha256_u32le='different'),
                       lambda c: c.update(max_tokens=32),
                       lambda c: c.update(prompt_format='chat:different'),
                       lambda c: c.update(cells=[])):
            candidate = copy.deepcopy(ref)
            mutate(candidate)
            with self.assertRaises(ValueError):
                compare_captures(ref, candidate)
        ref['cells'].pop()
        with self.assertRaises(ValueError):
            compare_captures(ref, ref)

    def test_joint_misses_are_inconclusive(self):
        ref = self.capture()
        for c in ref['cells']:
            c['text'] = 'wrong'
        self.assertEqual(compare_captures(ref, ref)['verdict'], 'inconclusive')
        passing = self.capture()
        self.assertEqual(compare_captures(passing, passing)['verdict'], 'pass')


if __name__ == "__main__":
    unittest.main()
