"""Contracts for the curated, machine-readable result catalog."""

import json
import unittest

from _bootstrap import ROOT


class ResultCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "results" / "manifest.json").read_text())

    def test_stage_ids_are_unique_and_follow_the_project_story(self):
        ids = [stage["id"] for stage in self.catalog["stages"]]
        self.assertEqual(ids, [
            "engine-foundations",
            "packed-prefill",
            "mixed-scheduling",
            "focused-optimization",
            "final-scorecard",
        ])

    def test_every_canonical_artifact_exists(self):
        artifacts = [
            artifact
            for stage in self.catalog["stages"]
            for artifact in stage["artifacts"]
        ]
        self.assertEqual(len(artifacts), len(set(artifacts)))
        missing = [artifact for artifact in artifacts if not (ROOT / artifact).exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
