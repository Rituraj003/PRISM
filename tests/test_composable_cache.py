import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.methods.composable import _load_cached_populations, _load_seed_population
from src.shared import ResultLogger


class TestComposableCache(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ResultLogger.header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_load_cached_populations_ignores_model_url(self) -> None:
        population_signature = {
            "dataset": "mock_dataset",
            "question_id": "question-1",
            "question_index": 0,
            "create_stage": "CreatePopulationStage",
            "pop_to_pop_stage": None,
            "depth_iterations": 0,
            "settings": {"temperature": 0.7},
        }
        cached_signature = {
            **population_signature,
            "settings": {
                "temperature": 0.7,
                "model_url": "https://old-host/v1",
                "verifier_model_url": "https://old-verifier/v1",
            },
        }

        metadata = {"response": {"population_signature": cached_signature}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "depth.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "dataset": population_signature["dataset"],
                        "question_id": population_signature["question_id"],
                        "question_index": population_signature["question_index"],
                        "run_id": "run-123",
                        "step": 0,
                        "chain_id": 0,
                        "reasoning": "Cached reasoning",
                        "raw_answer": "cached answer",
                        "total_input_tokens": 11,
                        "total_output_tokens": 7,
                        "metadata_json": json.dumps(metadata),
                    },
                    {
                        "dataset": population_signature["dataset"],
                        "question_id": population_signature["question_id"],
                        "question_index": population_signature["question_index"],
                        "run_id": "run-123",
                        "step": 1,
                        "chain_id": "",
                        "reasoning": "Final reasoning",
                        "raw_answer": "final answer",
                        "total_input_tokens": 11,
                        "total_output_tokens": 7,
                        "metadata_json": json.dumps(
                            {
                                "response": {
                                    "final": True,
                                    "population_signature": cached_signature,
                                }
                            }
                        ),
                    },
                ],
            )

            cached = _load_cached_populations(csv_path, population_signature)

        self.assertIsNotNone(cached)
        assert cached is not None
        populations, usage = cached
        self.assertEqual(len(populations), 1)
        self.assertEqual(len(populations[0]), 1)
        self.assertEqual(populations[0][0].choice, "cached answer")
        self.assertEqual(populations[0][0].reasoning, "Cached reasoning")
        self.assertEqual(usage, (11, 7, 0, 0))

    def test_load_seed_population_ignores_model_url(self) -> None:
        seed_signature = {
            "dataset": "mock_dataset",
            "question_id": "question-2",
            "question_index": 5,
            "create_stage": "CreatePopulationStage",
            "pop_to_pop_stage": None,
            "depth_iterations": 0,
            "settings": {
                "model_name": "mock-model",
                "width": 4,
                "samples": 8,
                "temperature": 0.2,
            },
        }
        cached_seed_signature = {
            **seed_signature,
            "settings": {
                "temperature": 0.2,
                "model_url": "https://old-host/v1",
                "verifier_model_url": "https://old-verifier/v1",
            },
        }
        metadata = {"response": {"seed_signature": cached_seed_signature}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "depth.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "dataset": seed_signature["dataset"],
                        "question_id": seed_signature["question_id"],
                        "question_index": seed_signature["question_index"],
                        "run_id": "run-456",
                        "step": 0,
                        "chain_id": 0,
                        "reasoning": "Seed reasoning",
                        "raw_answer": "seed answer",
                        "total_input_tokens": 3,
                        "total_output_tokens": 2,
                        "metadata_json": json.dumps(metadata),
                        "settings_json": json.dumps(
                            {
                                "create_population": "create_population_stage",
                                "model_name": "mock-model",
                                "width": 4,
                                "samples": 8,
                                "temperature": 0.2,
                            }
                        ),
                    },
                    {
                        "dataset": seed_signature["dataset"],
                        "question_id": seed_signature["question_id"],
                        "question_index": seed_signature["question_index"],
                        "run_id": "run-456",
                        "step": 1,
                        "chain_id": "",
                        "reasoning": "Final reasoning",
                        "raw_answer": "final answer",
                        "total_input_tokens": 3,
                        "total_output_tokens": 2,
                        "metadata_json": json.dumps(
                            {
                                "response": {
                                    "final": True,
                                    "seed_signature": cached_seed_signature,
                                }
                            }
                        ),
                        "settings_json": json.dumps(
                            {
                                "create_population": "create_population_stage",
                                "model_name": "mock-model",
                                "width": 4,
                                "samples": 8,
                                "temperature": 0.2,
                            }
                        ),
                    },
                ],
            )

            cached = _load_seed_population(csv_path, seed_signature)

        self.assertIsNotNone(cached)
        assert cached is not None
        answers, usage = cached
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0].choice, "seed answer")
        self.assertEqual(answers[0].reasoning, "Seed reasoning")
        self.assertEqual(usage, (3, 2, 0, 0))

    def test_cached_populations_skips_incomplete_run(self) -> None:
        population_signature = {
            "dataset": "mock_dataset",
            "question_id": "question-0",
            "question_index": 0,
            "create_stage": "CreatePopulationStage",
            "pop_to_pop_stage": None,
            "depth_iterations": 0,
            "settings": {"temperature": 0.7},
        }

        metadata_question0 = {
            "response": {"population_signature": population_signature}
        }
        metadata_question0_final = {
            "response": {"population_signature": population_signature, "final": True}
        }
        population_signature_q1 = {
            **population_signature,
            "question_id": "question-1",
            "question_index": 1,
        }
        metadata_question1 = {
            "response": {"population_signature": population_signature_q1}
        }
        metadata_question1_final = {
            "response": {
                "population_signature": population_signature_q1,
                "final": True,
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "depth.csv"
            self._write_csv(
                csv_path,
                [
                    # Complete run (run-001) with questions 0 and 1
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-0",
                        "question_index": 0,
                        "run_id": "run-001",
                        "step": 0,
                        "chain_id": 0,
                        "reasoning": "complete run",
                        "raw_answer": "complete answer",
                        "metadata_json": json.dumps(metadata_question0),
                    },
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-0",
                        "question_index": 0,
                        "run_id": "run-001",
                        "step": 1,
                        "chain_id": "",
                        "reasoning": "",
                        "raw_answer": "",
                        "metadata_json": json.dumps(metadata_question0_final),
                    },
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-1",
                        "question_index": 1,
                        "run_id": "run-001",
                        "step": 0,
                        "chain_id": 0,
                        "reasoning": "complete other question",
                        "raw_answer": "other answer",
                        "metadata_json": json.dumps(metadata_question1),
                    },
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-1",
                        "question_index": 1,
                        "run_id": "run-001",
                        "step": 1,
                        "chain_id": "",
                        "reasoning": "",
                        "raw_answer": "",
                        "metadata_json": json.dumps(metadata_question1_final),
                    },
                    # Partial run (run-999) only answered question 0
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-0",
                        "question_index": 0,
                        "run_id": "run-999",
                        "step": 0,
                        "chain_id": 0,
                        "reasoning": "partial run",
                        "raw_answer": "partial answer",
                        "metadata_json": json.dumps(metadata_question0),
                    },
                    {
                        "dataset": "mock_dataset",
                        "question_id": "question-0",
                        "question_index": 0,
                        "run_id": "run-999",
                        "step": 1,
                        "chain_id": "",
                        "reasoning": "",
                        "raw_answer": "",
                        "metadata_json": json.dumps(metadata_question0_final),
                    },
                ],
            )

            cached = _load_cached_populations(csv_path, population_signature)

        self.assertIsNotNone(cached)
        assert cached is not None
        populations, _ = cached
        self.assertEqual(populations[0][0].reasoning, "complete run")


if __name__ == "__main__":
    unittest.main()
