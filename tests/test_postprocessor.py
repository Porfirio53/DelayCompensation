"""Current vendor's trend, input packing and inverse-differencing regression checks."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def processor_module():
    # Import only the small postprocessor module, bypassing GPU/kernel imports.
    spec = importlib.util.spec_from_file_location(
        "_tuning_test_current", ROOT / "vendor/tirex/src/tirex2/model/component/postprocessor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_whole_history_trends_and_raw_rows_are_packed_independently(processor_module):
    import torch

    processor = processor_module.PostProcessor()
    t = torch.arange(192, dtype=torch.float32)
    targets = [torch.stack((t, torch.sin(t), torch.zeros_like(t))),
               torch.stack((-t[:80], torch.ones(80), torch.cos(t[:80])))]
    targets[0][0, :20] = torch.nan
    assert processor._detect_trends(targets[0]) == [True, False, False]
    batch, _, params = processor.transform_input(targets, 12, [None, None], [None, None])
    assert params["diff_masks"] == [[True, False, False], [True, False, False]]
    assert int(batch["target_mask"].sum()) == 6
    first = batch["x"][0, :192]
    finite = first[torch.isfinite(first)]
    assert len(finite) > 100
    assert finite[0] == 0  # The first value after the missing prefix starts a new difference chain.
    torch.testing.assert_close(finite[1:], torch.ones_like(finite[1:]))
    torch.testing.assert_close(batch["x"][1, :192], targets[0][1])
    torch.testing.assert_close(batch["x"][2, :192], targets[0][2])


def test_recent_trend_detected_independently_and_anchored_to_last_observation(processor_module):
    import torch

    recent = torch.cat((torch.zeros(160), torch.arange(32, dtype=torch.float32)))
    target = torch.stack((recent, torch.full((192,), 20.), -recent))
    assert processor_module.PostProcessor()._detect_trends(target) == [False, False, False]
    processor = processor_module.PostProcessor({"trend_window_points": 32})
    assert processor._detect_trends(target) == [True, False, True]
    assert processor_module.PostProcessor({"trend_window_points": 999})._detect_trends(target) == [False] * 3
    assert processor._detect_trends(torch.arange(3.)) == [False]
    rates = torch.full_like(target, 9.)
    clock = torch.arange(199.).unsqueeze(0)
    batch, _, params = processor.transform_input([target], 7, [rates], [clock])
    assert params["diff_masks"] == [[True, False, True]]
    torch.testing.assert_close(batch["x"][1, :192], target[1])
    torch.testing.assert_close(batch["x"][3:6, :192], rates)
    torch.testing.assert_close(batch["x"][6, :], clock[0])
    raw = torch.full((3, 9, 7), 0.25)
    restored = processor._diff_inverse(raw, torch.tensor([True, False, True]), processor._diff_params(target))
    torch.testing.assert_close(restored[0, 4], target[0, -1] + torch.arange(1., 8.) * 0.25)
    torch.testing.assert_close(restored[2, 4], target[2, -1] + torch.arange(1., 8.) * 0.25)
    torch.testing.assert_close(restored[1], raw[1])


@pytest.mark.parametrize("value", [True, 0, 3, 4.5])
def test_fixed_window_rejects_ambiguous_or_too_short_values(processor_module, value):
    with pytest.raises(ValueError, match="integer >= 4"):
        processor_module.PostProcessorConfig(trend_window_points=value)

