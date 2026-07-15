import json

import pytest

from src.labeling import fire_probability_from_binary_output, labels_from_class_indices, load_class_indices


def test_labels_follow_generator_indices():
    assert labels_from_class_indices({"no_fire": 1, "fire": 0}) == ["fire", "no_fire"]


def test_fire_probability_inverts_when_fire_is_class_zero():
    assert fire_probability_from_binary_output(0.8, {"fire": 0, "no_fire": 1}) == pytest.approx(0.2)


def test_fire_probability_is_direct_when_fire_is_class_one():
    assert fire_probability_from_binary_output(0.8, {"no_fire": 0, "fire": 1}) == 0.8


def test_load_class_indices_next_to_model(tmp_path):
    model_path = tmp_path / "model.keras"
    (tmp_path / "class_indices.json").write_text(json.dumps({"no_fire": 0, "fire": 1}), encoding="utf-8")
    assert load_class_indices(model_path) == {"no_fire": 0, "fire": 1}
