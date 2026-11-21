from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from datasets.schemas import DatasetRecord, DuplicatePair

NORMALIZATION_VERSION = "question-token-trigrams-v1"


def canonical_question(question: str) -> str:
    return " ".join(re.findall(r"\w+", question.casefold(), flags=re.UNICODE))


def question_shingles(question: str) -> frozenset[str]:
    tokens = re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
    if len(tokens) < 3:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2))


def find_duplicate_pairs(
    records: Sequence[DatasetRecord], threshold: float = 0.9
) -> list[DuplicatePair]:
    """Find exact-normalized or 3-token-shingle Jaccard overlaps at a set threshold.

    Prefix filtering orders shingles by corpus frequency and indexes only each
    item's Jaccard prefix. Candidate pairs are then checked with exact Jaccard;
    common short phrases therefore avoid a full all-pairs scan.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    shingles = [question_shingles(record.question) for record in records]
    frequencies = Counter(shingle for item in shingles for shingle in item)
    ordered = [
        sorted(item, key=lambda shingle: (frequencies[shingle], shingle)) for item in shingles
    ]
    index: dict[str, list[int]] = defaultdict(list)
    ordered_indices = sorted(
        range(len(records)), key=lambda i: (len(shingles[i]), records[i].sample_id)
    )
    found: dict[tuple[int, int], float] = {}
    for current in ordered_indices:
        candidates = {
            previous for shingle in ordered[current] for previous in index.get(shingle, ())
        }
        for previous in candidates:
            union = shingles[current] | shingles[previous]
            similarity = len(shingles[current] & shingles[previous]) / len(union) if union else 1.0
            exact = canonical_question(records[current].question) == canonical_question(
                records[previous].question
            )
            if exact or similarity >= threshold:
                found[(previous, current)] = similarity
        prefix_length = max(
            1, len(shingles[current]) - math.ceil(threshold * len(shingles[current])) + 1
        )
        for shingle in ordered[current][:prefix_length]:
            index[shingle].append(current)

    pairs = []
    for (left, right), similarity in sorted(
        found.items(),
        key=lambda item: (records[item[0][0]].sample_id, records[item[0][1]].sample_id),
    ):
        left_record, right_record = records[left], records[right]
        pairs.append(
            DuplicatePair(
                left_sample_id=left_record.sample_id,
                right_sample_id=right_record.sample_id,
                similarity=similarity,
                exact_question=canonical_question(left_record.question)
                == canonical_question(right_record.question),
                cross_split=left_record.assigned_split != right_record.assigned_split,
                left_split=left_record.assigned_split,
                right_split=right_record.assigned_split,
            )
        )
    return pairs


def split_training_records(
    records: Sequence[DatasetRecord],
    seed: int = 42,
    train_fraction: float = 0.9,
    threshold: float = 0.9,
) -> tuple[list[DatasetRecord], list[DuplicatePair]]:
    """Split only official training rows; keep near-duplicate groups together.

    Groups are connected components of exact/near-duplicate pairs. A seeded
    shuffle and greedy target-distance rule assign whole groups to validation.
    Official test rows are left in the test split and never used in the shuffle.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("sample_id values must be unique")
    training = [record for record in records if record.original_split == "train"]
    tests = [
        record.model_copy(update={"assigned_split": "test"})
        for record in records
        if record.original_split == "test"
    ]
    if len(training) < 2:
        raise ValueError(
            "at least two official training records are required to create train/validation"
        )

    training_pairs = find_duplicate_pairs(training, threshold)
    parent = list(range(len(training)))
    by_id = {record.sample_id: index for index, record in enumerate(training)}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in training_pairs:
        left, right = find(by_id[pair.left_sample_id]), find(by_id[pair.right_sample_id])
        if left != right:
            parent[right] = left
    groups: dict[int, list[DatasetRecord]] = defaultdict(list)
    for index, record in enumerate(training):
        groups[find(index)].append(record)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    validation_target = round(len(training) * (1 - train_fraction))
    validation_count = 0
    assigned: list[DatasetRecord] = []
    for group in group_list:
        goes_to_validation = abs(validation_target - (validation_count + len(group))) < abs(
            validation_target - validation_count
        )
        target_split = "validation" if goes_to_validation else "train"
        if goes_to_validation:
            validation_count += len(group)
        assigned.extend(
            record.model_copy(update={"assigned_split": target_split}) for record in group
        )
    if not any(record.assigned_split == "validation" for record in assigned):
        raise ValueError(
            "duplicate grouping left no validation examples; use a larger training source"
        )
    result = sorted([*assigned, *tests], key=lambda record: record.sample_id)
    return result, training_pairs
