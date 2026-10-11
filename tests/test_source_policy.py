"""Reject unsupported sources before any model, reference or dataset is opened."""
import json
from types import SimpleNamespace

import pytest

from mytirex.aoi_experiment import eval_aoi
from mytirex.cli import main
from mytirex.common import current_source, publish_success
from mytirex.evaluation import resolved_config
from mytirex.model import TiRexModel, read_tuning
from mytirex.references import reference_identity


@pytest.mark.parametrize("command", ["doctor", "eval-speed", "benchmark", "eval-aoi"])
def test_cli_rejects_unsupported_source_before_dispatch(command, tmp_path):
    with pytest.raises(SystemExit) as result:
        main([command, "--root", str(tmp_path), "--source", "official"])
    assert result.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["official", "other"])
def test_python_entry_points_reject_unsupported_sources_before_io(source, tmp_path):
    with pytest.raises(ValueError, match="Only the current TiRex source"):
        TiRexModel(tmp_path, {}, {}, source)
    with pytest.raises(ValueError, match="Only the current TiRex source"):
        read_tuning(tmp_path, "missing.json", source)
    with pytest.raises(ValueError, match="Only the current TiRex source"):
        resolved_config(tmp_path, SimpleNamespace(source=source))
    with pytest.raises(ValueError, match="Only the current TiRex source"):
        eval_aoi(tmp_path, SimpleNamespace(source=source))
    assert not list(tmp_path.iterdir())


def test_current_model_cannot_select_a_different_source_id(tmp_path):
    with pytest.raises(ValueError, match="requires source_directory='TiRex'"):
        TiRexModel(tmp_path, {"source_directory": "baselines/tirex-2"}, {})


def test_current_source_index_cannot_redirect_to_another_vendor(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "source_index.json").write_text(json.dumps({"TiRex": {"directory": "vendor/tirex_official"}}))
    with pytest.raises(ValueError, match="must resolve to vendor/tirex"):
        current_source(tmp_path)


def test_unsupported_reference_is_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="Only current-source historical predictions"):
        reference_identity(tmp_path, "official_speed", {})
    assert not list(tmp_path.iterdir())


def test_unsupported_source_cannot_become_the_latest_result(tmp_path):
    output = tmp_path / "archived"
    output.mkdir()
    (output / "metadata.json").write_text(json.dumps({"status": "success", "model": {"source": "official"}}))
    with pytest.raises(ValueError, match="Only the current TiRex source"):
        publish_success(tmp_path, output, "aoi")
    assert not (tmp_path / "outputs/latest_aoi.json").exists()
