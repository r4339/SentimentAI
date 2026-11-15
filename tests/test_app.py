import unittest

from app import normalize_label, predict_review


class FakePipeline:
    classes_ = ["negative", "positive"]

    def predict(self, reviews):
        self.seen = reviews
        return ["positive"]

    def predict_proba(self, reviews):
        return [[0.2, 0.8]]


class SentimentHelpersTests(unittest.TestCase):
    def test_normalizes_supported_labels(self):
        self.assertEqual(normalize_label("pos"), "Positive")
        self.assertEqual(normalize_label(0), "Negative")

    def test_custom_label_map(self):
        self.assertEqual(normalize_label("good", {"good": "Positive"}), "Positive")

    def test_unknown_label_is_explained(self):
        with self.assertRaisesRegex(ValueError, "LABEL_MAP"):
            normalize_label("neutral")

    def test_predicts_raw_text_and_returns_predicted_class_probability(self):
        model = FakePipeline()
        result = predict_review(model, "A very good product")
        self.assertEqual(model.seen, ["A very good product"])
        self.assertEqual(result, ("Positive", 0.8))


if __name__ == "__main__":
    unittest.main()
