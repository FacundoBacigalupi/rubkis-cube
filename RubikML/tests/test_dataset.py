import torch
from rubikml.dataset import RubikScrambleDataset
from rubikml.moves import MOVES


def test_dataset_yields_correct_shapes():
    ds = RubikScrambleDataset(min_depth=1, max_depth=3, samples_per_epoch=20, seed=1)
    for x, y, v in ds:
        assert x.shape == (324,)
        assert x.dtype == torch.float32
        assert y.shape == ()
        assert 0 <= y.item() < len(MOVES)
        assert v.dtype == torch.float32
        assert 0.0 < v.item() <= 1.0


def test_dataset_is_reproducible():
    def collect(seed):
        ds = RubikScrambleDataset(1, 3, 10, seed=seed)
        return [(x.tolist(), y.item(), v.item()) for x, y, v in ds]

    assert collect(42) == collect(42)
    assert collect(42) != collect(99)


def test_dataset_covers_all_labels():
    ds = RubikScrambleDataset(1, 1, 500, seed=7)
    labels = {y.item() for _, y, _ in ds}
    # depth-1 from solved: all 12 moves can appear as labels
    assert len(labels) == len(MOVES)


def test_value_target_normalized():
    max_depth = 5
    ds = RubikScrambleDataset(1, max_depth, 100, seed=3)
    for _, _, v in ds:
        assert 0.0 < v.item() <= 1.0
