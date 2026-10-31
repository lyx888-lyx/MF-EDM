"""Standard-library tests: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_data import check_files, main  # noqa: E402


class FeaturePreflightTests(unittest.TestCase):
    def test_missing_iemocap_files(self):
        with tempfile.TemporaryDirectory() as directory:
            checked, missing = check_files(directory, "IEMOCAP")
            self.assertEqual(len(checked), 2)
            self.assertEqual(len(missing), 2)

    def test_present_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "M3ED_features" / "M3ED_features.pkl"
            path.parent.mkdir()
            path.write_bytes(b"fixture")
            checked, missing = check_files(directory, "M3ED")
            self.assertEqual(missing, [])
            self.assertTrue(checked[0][2])
            self.assertEqual(main(["--dataset", "M3ED", "--root", directory]), 0)

    def test_empty_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "KMIND_features" / "KMIND_features_normalized.pkl"
            path.parent.mkdir()
            path.touch()
            _, missing = check_files(directory, "KMIND")
            self.assertEqual(len(missing), 1)


if __name__ == "__main__":
    unittest.main()
