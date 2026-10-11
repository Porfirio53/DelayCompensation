"""Locate the local CUDA toolchain without requiring environment activation.

TiRex's FlashRNN CUDA kernels are compiled on first use.  The CUDA runtime in a
PyTorch wheel alone is insufficient: nvcc and development headers are required.
This helper is also usable before importing the standalone ``tirex2`` package.
"""

import os
from pathlib import Path
import shutil
import sys
import sysconfig
import subprocess


def configure_cuda_toolchain(*, required: bool = True) -> None:
    """Respect an explicit CUDA_HOME; otherwise prefer the active Python prefix."""
    env_bin = str(Path(sys.prefix) / "bin")
    search_path = os.environ.get("PATH", "").split(os.pathsep)
    if env_bin not in search_path:
        os.environ["PATH"] = os.pathsep.join([env_bin, *search_path])

    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit:
        cuda_home = Path(explicit)
    else:
        nvcc = shutil.which("nvcc")
        candidates = [Path(sys.prefix), Path("/usr/local/cuda")]
        if nvcc:
            candidates.append(Path(nvcc).resolve().parents[1])
        cuda_home = next((path for path in candidates if (path / "bin/nvcc").is_file()), None)
        if cuda_home is None and not required:
            # xLSTM inspects CUDA includes even for CPU loading on a GPU host.
            runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime"
            if (runtime / "include").is_dir():
                cuda_home = runtime

    if required and (cuda_home is None or not (cuda_home / "bin/nvcc").is_file()):
        raise RuntimeError("TiRex-2 GPU inference requires a CUDA toolkit with nvcc; "
                           "install CUDA 12.8 in .venv or set CUDA_HOME to its installation")
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        # Conda CUDA packages keep headers under targets/<arch>/include rather
        # than <prefix>/include.  Both dependencies expose an include override.
        target_includes = [str(path) for path in sorted((cuda_home / "targets").glob("*/include"))
                           if (path / "cuda_runtime.h").is_file()]
        # Torch's C++ headers also include cuSPARSE/cuSOLVER headers, which are
        # already installed with its NVIDIA wheels in this environment.
        nvidia_wheels = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        target_includes.extend(str(path) for path in sorted(nvidia_wheels.glob("*/include"))
                               if path.is_dir())
        for key in ("FLASHRNN_EXTRA_INCLUDE_PATHS", "XLSTM_EXTRA_INCLUDE_PATHS"):
            includes = [path for path in os.environ.get(key, "").split(os.pathsep) if path]
            includes.extend(path for path in target_includes if path not in includes)
            if includes:
                os.environ[key] = os.pathsep.join(includes)
    os.environ.setdefault("MAX_JOBS", "4")


def configure_runtime(root: Path) -> None:
    """Give this deployment its own kernel caches before importing torch/xLSTM."""
    configure_cuda_toolchain(required=True)
    cache = root / ".cache"
    for key, directory in (("TORCH_EXTENSIONS_DIR", cache / "torch_extensions"),
                           ("TRITON_CACHE_DIR", cache / "triton"),
                           ("TORCHINDUCTOR_CACHE_DIR", cache / "inductor")):
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    # FlashRNN adds -lcublas. Conda's CUDA development libraries live under
    # targets as well as lib; expose both to the host linker, without old paths.
    cuda = Path(os.environ["CUDA_HOME"])
    libraries = [p for p in (cuda / "lib", cuda / "lib64", *sorted((cuda / "targets").glob("*/lib")))
                 if p.is_dir()]
    wheels = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libraries.extend(p for p in sorted(wheels.glob("*/lib")) if p.is_dir())
    previous = [p for p in os.environ.get("LIBRARY_PATH", "").split(os.pathsep) if p]
    os.environ["LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys([*(str(p) for p in libraries), *previous]))


def cache_snapshot(root: Path) -> dict:
    files = [p for name in ("torch_extensions", "triton", "inductor")
             for p in (root / ".cache" / name).rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def toolchain_report() -> dict:
    cuda = Path(os.environ["CUDA_HOME"])
    nvcc = cuda / "bin/nvcc"
    version = subprocess.check_output([str(nvcc), "--version"], text=True, timeout=15).strip()
    includes = os.environ.get("FLASHRNN_EXTRA_INCLUDE_PATHS", "").split(os.pathsep)
    found_headers = {name: [str(Path(p) / name) for p in includes if p and (Path(p) / name).is_file()]
                     for name in ("cuda.h", "cuda_runtime.h", "cublas_v2.h", "cusparse.h", "cusolverDn.h")}
    if any(not paths for paths in found_headers.values()):
        raise RuntimeError(f"CUDA development headers are incomplete: {found_headers}")
    link_dirs = os.environ.get("LIBRARY_PATH", "").split(os.pathsep)
    cublas = [str(Path(p) / "libcublas.so") for p in link_dirs if (Path(p) / "libcublas.so").is_file()]
    if not cublas:
        raise RuntimeError("cuBLAS development link library libcublas.so is missing")
    return {"python_prefix": sys.prefix, "conda_prefix_environment": (Path(sys.prefix) / "conda-meta").is_dir(),
            "nvcc": str(nvcc), "nvcc_version": version, "CUDA_HOME": str(cuda),
            "FLASHRNN_EXTRA_INCLUDE_PATHS": os.environ.get("FLASHRNN_EXTRA_INCLUDE_PATHS"),
            "XLSTM_EXTRA_INCLUDE_PATHS": os.environ.get("XLSTM_EXTRA_INCLUDE_PATHS"),
            "MAX_JOBS": os.environ["MAX_JOBS"], "headers": found_headers,
            "LIBRARY_PATH": os.environ["LIBRARY_PATH"], "cublas_link_libraries": cublas,
            "kernel_cache_directories": {k: os.environ.get(k) for k in
                                         ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}}
