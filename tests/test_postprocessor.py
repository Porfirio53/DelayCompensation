"""Retained vendor postprocessor equivalence, trend and inverse-differencing checks."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def processors():
    # Import only the small postprocessor module, bypassing GPU/kernel imports.
    modules = []
    for name, directory in (("official", "vendor/tirex_official"), ("optimized", "vendor/tirex")):
        spec = importlib.util.spec_from_file_location(
            f"_tuning_test_{name}", ROOT / directory / "src/tirex2/model/component/postprocessor.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_default_root_copy_preserves_official_transform(processors):
    import torch

    official, optimized = processors
    original, copied = official.PostProcessor(), optimized.PostProcessor()
    assert asdict(copied.cfg) == {**asdict(original.cfg), "trend_window_points": None}
    t = torch.arange(192, dtype=torch.float32)
    targets = [torch.stack((t, torch.sin(t), torch.zeros_like(t))),
               torch.stack((-t[:80], torch.ones(80), torch.cos(t[:80])))]
    targets[0][0, :20] = torch.nan
    assert copied._detect_trends(targets[0]) == original._detect_trends(targets[0])
    inputs = [processor.transform_input(targets, 12, [None, None], [None, None])
              for processor in (original, copied)]
    for name in inputs[0][0]:
        torch.testing.assert_close(inputs[0][0][name], inputs[1][0][name], rtol=0, atol=0, equal_nan=True)
    assert inputs[0][2]["diff_masks"] == inputs[1][2]["diff_masks"]


def test_recent_trend_detected_independently_and_anchored_to_last_observation(processors):
    import torch

    official, optimized = processors
    recent = torch.cat((torch.zeros(160), torch.arange(32, dtype=torch.float32)))
    target = torch.stack((recent, torch.full((192,), 20.), -recent))
    assert official.PostProcessor()._detect_trends(target) == [False, False, False]
    processor = optimized.PostProcessor({"trend_window_points": 32})
    assert processor._detect_trends(target) == [True, False, True]
    assert optimized.PostProcessor({"trend_window_points": 999})._detect_trends(target) == [False] * 3
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
def test_fixed_window_rejects_ambiguous_or_too_short_values(processors, value):
    with pytest.raises(ValueError, match="integer >= 4"):
        processors[1].PostProcessorConfig(trend_window_points=value)

