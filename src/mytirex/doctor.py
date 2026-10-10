"""Environment/content diagnostics and actual GPU trajectory validation."""
from pathlib import Path
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (atomic_json, dependency_versions, integration_identity, new_output, project_path,
                     publish_success, read_json, sha256, verify_migration)
from .data import dataset_identity, load_dataset
from .evaluation import resolved_config
from .gpu import configure_runtime, toolchain_report
from .model import TiRexModel
from .prepared import check_examples, verify_prepared
from .references import reference_identity


def extension_links(root: Path) -> list[dict]:
    import torch
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(Path(torch.__file__).parent / "lib"), os.environ.get("LIBRARY_PATH", ""),
        os.environ.get("LD_LIBRARY_PATH", "")])
    result = []
    for file in sorted((root / ".cache/torch_extensions").rglob("*.so")):
        links = subprocess.check_output(["ldd", str(file)], env=environment, text=True, timeout=15)
        if "not found" in links:
            raise RuntimeError(f"Compiled extension has unresolved libraries: {file}\n{links}")
        result.append({"file": str(file), "sha256": sha256(file), "links": links.strip()})
    return result


def doctor(root: Path, args) -> Path:
    resolved = resolved_config(root, args, batch_size=1)
    output = new_output(root, f"doctor_{args.source}")
    atomic_json(output / "config.json", resolved)
    try:
        sequences = load_dataset(root)
        data_index = read_json(root / "configs/data_index.json")
        prepared_path = project_path(root, data_index["prepared_comma2k19_v1"]["directory"])
        prepared = verify_prepared(prepared_path)
        if prepared["dataset_id"] != data_index["prepared_comma2k19_v1"]["dataset_id"]:
            raise ValueError("Prepared corpus index identity mismatch")
        _, official = reference_identity(root, "official_speed", resolved["experiment"],
                                         resolved["settings"], resolved["runtime"])
        _, current = reference_identity(root, "current_speed", resolved["experiment"],
                                        resolved["settings"], resolved["runtime"])
        configure_runtime(root)
        environment = toolchain_report()
        import torch
        if sys.version_info[:2] != (3, 12) or not environment["conda_prefix_environment"]:
            raise RuntimeError("Use this project's Python 3.12 Conda prefix environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        if not torch.__version__.startswith("2.8.0") or torch.version.cuda != "12.8" or "release 12.8" not in environment["nvcc_version"]:
            raise RuntimeError("The fixed baseline requires PyTorch 2.8.0 and CUDA/nvcc 12.8")
        environment.update({"torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                            "gpu_name": torch.cuda.get_device_name(), "gpu_capability": list(torch.cuda.get_device_capability()),
                            "executable": sys.executable})
        model = TiRexModel(root, resolved["settings"], resolved["runtime"], args.source)
        checks = {"regression_sequences": len(sequences), "points_per_sequence": len(sequences[0].values),
                  "prepared_dataset_id": prepared["dataset_id"], "prepared_summary": prepared["summary"],
                  "prepared_examples": check_examples(prepared_path),
                  "official_reference": official, "current_reference": current,
                  "weight_content_verified": True, "selected_source_sha256": model.source_sha256}
        if args.verify_assets:
            checks["migration"] = verify_migration(root)
        metadata = {"status": "success", "kind": "doctor", "environment": environment, "checks": checks,
                    "dependencies": dependency_versions(), "integration": integration_identity(root),
                    "dataset": dataset_identity(root), "smoke_requested": args.smoke,
                    "gpu_inference_validated": False}
        if args.smoke:
            print(f"Loading {args.source} TiRex for real GPU smoke; output={output}", flush=True)
            model.load()
            rows, timings = [], []
            for delay in (175, 300):
                started = perf_counter()
                view = sequences[0].request(200, delay * 1000)
                trajectory = model.forecast(view, delay * 1000 // 25000)
                point = trajectory.value_at(view.query_time_us)
                if trajectory.status != "ok" or not np.isfinite(point):
                    raise RuntimeError(f"GPU smoke prediction failed: {trajectory.failure_reason}")
                if trajectory.values_mps.shape != (9, delay * 1000 // 25000):
                    raise RuntimeError("Unexpected GPU trajectory shape")
                timings.append({"delay_ms": delay, "horizon_steps": trajectory.values_mps.shape[1],
                                "end_to_end_s": perf_counter() - started, "point_mps": point,
                                "parameter_devices": model.parameter_devices})
                for step, timestamp in enumerate(trajectory.future_time_us):
                    rows.append({"delay_ms": delay, "source_time_us": trajectory.source_time_us,
                                 "future_time_us": int(timestamp), "step": step + 1,
                                 **{f"q_{float(q):.1f}": float(trajectory.values_mps[index, step])
                                    for index, q in enumerate(trajectory.quantiles)}})
                print(f"GPU smoke {delay}ms: shape={trajectory.values_mps.shape}, median={point:.7f} m/s", flush=True)
            pd.DataFrame(rows).to_csv(output / "smoke_forecast.csv", index=False)
            metadata.update({"gpu_inference_validated": True, "smoke": timings, "model": model.metadata(),
                             "compiled_extension_links": extension_links(root)})
        atomic_json(output / "metadata.json", metadata)
        from .report import write_json_report
        write_json_report(output, "TiRex-2 环境与 GPU 验证", {"gpu_inference_validated": metadata["gpu_inference_validated"],
                          "environment": environment, "checks": checks, "smoke": metadata.get("smoke")})
        publish_success(root, output, "doctor")
        print(f"Diagnostics complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "gpu_inference_validated": False,
                                               "error": f"{type(exc).__name__}: {exc}"})
        raise
