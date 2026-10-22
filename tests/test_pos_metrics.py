import pytest
from pos_metrics import compute_pos_metrics


def test_plain_pos_tags_use_token_metrics():
    result = compute_pos_metrics([["NOUN", "NOUN", "VERB"]], [["NOUN", "VERB", "VERB"]], True)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["NOUN_precision"] == 1
    assert result["NOUN_recall"] == .5
    assert result["VERB_precision"] == .5
    assert result["VERB_recall"] == 1


def test_sentence_boundaries_do_not_create_chunks():
    a = compute_pos_metrics([["NOUN", "NOUN"]], [["NOUN", "VERB"]])
    b = compute_pos_metrics([["NOUN"], ["NOUN"]], [["NOUN"], ["VERB"]])
    assert a == b


def test_missing_class_and_no_tokens():
    result = compute_pos_metrics([["NOUN"]], [["VERB"]], True)
    assert result["f1"] == result["macro_f1"] == 0
    assert result["VERB_support"] == 0
    assert compute_pos_metrics([[]], [[]])["accuracy"] == 0


@pytest.mark.parametrize("gold,pred", [([["NOUN"]], []), ([["NOUN"]], [[]]), ([[1]], [["NOUN"]])])
def test_invalid_alignment(gold, pred):
    with pytest.raises(ValueError): compute_pos_metrics(gold, pred)


def test_imbalanced_scores_are_distinct():
    result = compute_pos_metrics([["NOUN"] * 9 + ["VERB"]], [["NOUN"] * 10])
    assert result["accuracy"] == .9
    assert result["macro_f1"] == pytest.approx((18 / 19) / 2)
    assert result["weighted_f1"] == pytest.approx((18 / 19) * .9)


def test_actual_trainer_metrics_ignore_special_and_subword_positions():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'run_ner.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_metrics')
    namespace = dict(np=np, label_list=['NOUN', 'VERB'], compute_pos_metrics=compute_pos_metrics,
                     data_args=SimpleNamespace(return_entity_level_metrics=True))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-metrics>', 'exec'), namespace)
    # Wrong predictions at ignored positions must not change the denominator.
    scores = np.array([[[0, 1], [1, 0], [0, 1], [0, 1], [1, 0]]])
    labels = np.array([[-100, 0, -100, 0, 1]])
    result = namespace['compute_metrics']((scores, labels))
    assert result['accuracy'] == pytest.approx(1 / 3)
    assert result['NOUN_support'] == 2
    assert result['VERB_support'] == 1
