import os
import sys
import unittest

_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from answer_verification import (
    check_answer_correctness,
    compute_prm_score,
    extract_mcq_answer,
    extract_prm_step_scores,
    normalize_answer,
)
from shared import QuestionType


class TestAnswerVerification(unittest.TestCase):
    def test_extract_mcq_answer(self):
        test_cases = [
            ("Some reasoning here\n<answer>a</answer>", "a"),
            ("Some reasoning\n<answer>\\boxed{b}</answer>", "b"),
            ("Therefore the answer is \\boxed{c}", "c"),
            ("After analysis, I choose d.", "d"),
            ("The answer is <answer>C</answer>", "c"),
            ("<answer> b </answer>", "b"),
            (
                "Consider options a and b. After analysis, c is wrong. <answer>d</answer>",
                "d",
            ),
        ]
        for text, expected in test_cases:
            with self.subTest(text=text):
                self.assertEqual(extract_mcq_answer(text), expected)

    def test_normalize_answer(self):
        test_cases = [
            ("a", QuestionType.MCQ, "a"),
            ("A", QuestionType.MCQ, "a"),
            ("b ", QuestionType.MCQ, "b"),
            ("42", QuestionType.MATH, "42"),
            ("42.0", QuestionType.MATH, "42"),
            ("-0.7", QuestionType.MATH, "-0.7"),
            ("\\frac{1}{2}", QuestionType.MATH, "1/2"),
        ]
        for text, question_type, expected in test_cases:
            with self.subTest(text=text, question_type=question_type):
                self.assertEqual(
                    normalize_answer(text, question_type=question_type), expected
                )

    def test_check_answer_correctness_formatting(self):
        test_cases = [
            (
                r"\sqrt{23}-2\sqrt3",
                r"\sqrt{23}-2 \sqrt{3}",
                True,
            ),
            ("0.5", r"\frac{1}{2}", True),
            ("0.333333", r"\frac{1}{3}", True),
            (r"\sqrt3", r"\sqrt{3}", True),
            (r"\frac12", r"\frac{1}{2}", True),
            (r"\frac{1}2", r"\frac{1}{2}", True),
            (r"\frac1{2}", r"\frac{1}{2}", True),
            (r"ab=14+4\sqrt{37}", r"14+4\sqrt{37}", True),
            (r"26!2^{25}", r"2^{25}\cdot 26!", True),
            (r"\frac{1}{2},\frac{3}{4}", r"\frac{3}{4},\frac{1}{2}", True),
        ]
        for predicted, correct, expected in test_cases:
            with self.subTest(predicted=predicted, correct=correct):
                self.assertEqual(
                    check_answer_correctness(
                        predicted, correct, question_type=QuestionType.MATH
                    ),
                    expected,
                )

    def test_prm_scoring_parses_tail_tokens(self):
        feedback = "\n".join(
            [
                '<step i="1">first step ok</step> +1',
                '<step i="2">second step wrong</step> -1',
                "<step>FINAL ANSWER CHECK: -1</step>",
                "<answer>-1</answer>",
            ]
        )
        self.assertEqual(extract_prm_step_scores(feedback), [1, -1])
        # step_ratio = (1 + 0.5*0) / 2 = 0.5; final=-1 => 0.5*0.3 = 0.15
        self.assertAlmostEqual(compute_prm_score(feedback), 0.15, places=6)

    def test_prm_scoring_ignores_literal_format_options(self):
        # Sometimes the verifier mistakenly copies the format spec ("+1|0|-1").
        # We should not mis-parse that as a real "-1" verdict.
        feedback = "\n".join(
            [
                '<step i="1">format copied +1|0|-1</step>',
                "<step>FINAL ANSWER CHECK: -1</step>",
                "<answer>-1</answer>",
            ]
        )
        self.assertEqual(extract_prm_step_scores(feedback), [None])
        # No usable step verdicts => neutral step_ratio=0.5; final=-1 => 0.15
        self.assertAlmostEqual(compute_prm_score(feedback), 0.15, places=6)

    def test_prm_scoring_handles_unicode_minus(self):
        # Some models emit U+2212 (minus sign) instead of ASCII '-'.
        # Previously this could be mis-parsed as a '+1' verdict.
        feedback = "\n".join(
            [
                '<step i="1">bad step \u22121</step>',
                "<step>FINAL ANSWER CHECK: -1</step>",
                "<answer>-1</answer>",
            ]
        )
        self.assertEqual(extract_prm_step_scores(feedback), [-1])
        self.assertAlmostEqual(compute_prm_score(feedback), 0.0, places=6)

        # Also handle the (disallowed-but-seen) case where the verdict token appears
        # *after* </step>, but uses a Unicode minus.
        feedback_tail = "\n".join(
            [
                '<step i="1">bad step</step> \u22121',
                "<step>FINAL ANSWER CHECK: -1</step>",
                "<answer>-1</answer>",
            ]
        )
        self.assertEqual(extract_prm_step_scores(feedback_tail), [-1])
        self.assertAlmostEqual(compute_prm_score(feedback_tail), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
