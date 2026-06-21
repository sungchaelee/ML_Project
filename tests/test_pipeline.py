import sys
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression

from preprocessing import load_raw_data, clean_data, TARGET
from train import make_pipeline
from inference import validate_input_ranges


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 테스트용 모델을 데이터로 한 번만 학습
        df = clean_data(load_raw_data())
        cls.X = df.drop(columns=TARGET)
        cls.y = df[TARGET]
        cls.model = make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42), scale=True
        )
        cls.model.fit(cls.X, cls.y)
        cls.sample = cls.X.head(10).copy()

    def test_prediction_shape(self):
        """예측 개수가 입력 행 수와 일치하는가."""
        preds = self.model.predict(self.sample)
        self.assertEqual(preds.shape[0], self.sample.shape[0])

    def test_probabilities_valid(self):
        """예측 확률이 [0,1] 범위이고 행마다 합이 1인가."""
        proba = self.model.predict_proba(self.sample)
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_input_range_validation(self):
        """임상 범위(chol 0~600)를 벗어난 입력을 잡아내는가."""
        bad = self.sample.copy()
        bad.loc[bad.index[0], "chol"] = 9999  # 범위 밖 값
        issues = validate_input_ranges(bad)
        self.assertIn("chol", issues)

    def test_determinism(self):
        """같은 입력 → 같은 출력 (고정 시드)."""
        p1 = self.model.predict(self.sample)
        p2 = self.model.predict(self.sample)
        np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()