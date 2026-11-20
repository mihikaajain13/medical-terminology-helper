import unittest
import subprocess
import sys
import numpy as np
from app.generate import Helper
from app.model import forward


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = Helper()

    def test_round_trip(self):
        tok = self.helper.tokenizer
        text = "Arteries, veins & café"
        self.assertEqual(tok.decode(tok.encode(text)), text.lower())

    def test_causal_mask(self):
        p = self.helper.weights
        a, _ = forward(p, np.array([[256, 4, 5, 6]]))
        b, _ = forward(p, np.array([[256, 4, 8, 9]]))
        np.testing.assert_allclose(a[:, :2], b[:, :2])

    def test_checkpoint(self):
        for prompt, expected in [
            ("Arteries carry", "Arteries carry blood away from the heart."),
            ("The right ventricle", "The right ventricle pumps blood to the lungs."),
            ("Bone marrow", "Bone marrow produces blood cells.")]:
            with self.subTest(prompt=prompt):
                self.assertEqual(self.helper.complete(prompt), expected)

    def test_empty(self):
        with self.assertRaises(ValueError):
            self.helper.complete("  ")

    def test_cli(self):
        result = subprocess.check_output(
            [sys.executable, "-m", "app.cli", "The left atrium"], text=True)
        self.assertEqual(result.strip(), "The left atrium receives blood from the lungs.")
