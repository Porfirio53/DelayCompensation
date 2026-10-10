"""Content identities and project-relative paths, extracted from the old helpers."""
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hash(path: Path) -> str:
    # Same relative-file content identity as experiment/cache.py. No path relocation
    # changes it, and bytecode / repository metadata never enter it.
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*"))
             if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
             and not p.name.endswith((".pyc", ".DS_Store"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def verify_files(directory: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory.resolve()) or not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Content checksum/path mismatch: {file}")


def dependency_versions() -> dict:
    return dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions()
                       if d.metadata["Name"]))


def new_output(root: Path, kind: str) -> Path:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    directory = root / "outputs" / f"{now:%Y%m%d_%H%M%S_%f}_{kind}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def publish_success(root: Path, output: Path, kind: str) -> None:
    atomic_json(root / "outputs" / f"latest_{kind}.json",
                {"directory": str(output.relative_to(root)), "status": "success", "kind": kind})


def integration_identity(root: Path) -> dict:
    return {"source_sha256": source_hash(root / "src/mytirex"),
            "pyproject_sha256": sha256(root / "pyproject.toml"),
            "runtime_lock_sha256": sha256(root / "configs/requirements-runtime.lock")}


def assert_local_imports(root: Path) -> dict:
    """Audit the active packages, including editable source and native dependencies."""
    result = {}
    for name in ("mytirex", "tirex2", "torch", "numpy", "flashrnn", "xlstm", "mlstm_kernels"):
        module = sys.modules.get(name)
        if module is None:
            continue
        file = Path(module.__file__).resolve()
        expected = root if name in {"mytirex", "tirex2"} else Path(sys.prefix)
        if not file.is_relative_to(expected.resolve()):
            raise RuntimeError(f"Nonlocal {name} import: {file}; expected it under {expected}")
        result[name] = str(file)
    return result


def verify_migration(root: Path) -> dict:
    """Verify destination content only; the historical source need not exist."""
    manifest = read_json(root / "configs/migration_manifest.json")
    for entry in manifest["files"]:
        path = project_path(root, entry["path"])
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Migration content changed: {entry['path']}")
    return {"files": len(manifest["files"]), "bytes": sum(e["bytes"] for e in manifest["files"]),
            "source_needed_at_runtime": False, "content_verified": True}
