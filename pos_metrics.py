"""Token classification metrics for plain Universal POS tags (not BIO chunks)."""

from collections import Counter


def compute_pos_metrics(references, predictions, include_per_tag=False):
    """Compute micro, macro and weighted scores after ignored tokens are removed.

    ``precision``, ``recall`` and ``f1`` are micro-averaged, matching token
    accuracy for single-label classification. No BIO/entity interpretation is
    applied to tags such as NOUN or VERB. Undefined ratios are reported as zero.
    """
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction sentence counts differ")
    support, predicted, correct = Counter(), Counter(), Counter()
    for index, (truth, guess) in enumerate(zip(references, predictions)):
        if len(truth) != len(guess):
            raise ValueError("Token counts differ in sentence {}".format(index))
        for gold, pred in zip(truth, guess):
            if not isinstance(gold, str) or not isinstance(pred, str):
                raise ValueError("POS tags must be strings")
            support[gold] += 1
            predicted[pred] += 1
            if gold == pred:
                correct[gold] += 1
    total = sum(support.values())
    accuracy = sum(correct.values()) / total if total else 0.0
    result = dict(accuracy=accuracy, precision=accuracy, recall=accuracy, f1=accuracy)
    per_tag = {}
    for tag in sorted(support.keys() | predicted.keys()):
        precision = correct[tag] / predicted[tag] if predicted[tag] else 0.0
        recall = correct[tag] / support[tag] if support[tag] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tag[tag] = dict(precision=precision, recall=recall, f1=f1, support=support[tag])
    result["macro_f1"] = sum(row["f1"] for row in per_tag.values()) / len(per_tag) if per_tag else 0.0
    result["weighted_f1"] = sum(row["f1"] * row["support"] for row in per_tag.values()) / total if total else 0.0
    if include_per_tag:
        for tag, row in per_tag.items():
            result.update({"{}_{}".format(tag, metric): value for metric, value in row.items()})
    return result
