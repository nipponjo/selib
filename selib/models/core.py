import hashlib
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from ..urls import HFHUB_BASE_URL, MODEL_URLS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_url(url: str) -> str:
    """Expand a registry URL/path into a downloadable URL."""
    if url.startswith(("http://", "https://", "file://")):
        return url
    return f"{HFHUB_BASE_URL}/resolve/main/{url.lstrip('/')}"


def _cache_relative_path(url: str) -> Path:
    """Return a stable cache path for a registry URL/path."""
    if url.startswith(("http://", "https://", "file://")):
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        return Path(name)
    return Path(url)


def get_model_cache_dir() -> Path:
    """Return the default directory used for downloaded ONNX models."""
    return Path.home() / ".cache" / "selib" / "models"


def _format_bytes(n_bytes: int) -> str:
    """Format a byte count for compact progress messages."""
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def download_file(url: str,
                  output_path: Path,
                  chunk_size: int = 1024 * 1024,
                  verbose: bool = True,
                  label: Optional[str] = None) -> None:
    """Download a URL to ``output_path`` using only the standard library."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    name = label or output_path.name
    if verbose:
        print(f"Downloading {name} ...")
        print(f"  from: {url}")
        print(f"  save to: {output_path}")

    with urlopen(url) as response, tmp_path.open("wb") as file:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        last_percent = -1
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if verbose:
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent != last_percent:
                        print(
                            f"\r  progress: {_format_bytes(downloaded)} / "
                            f"{_format_bytes(total)} ({percent}%)",
                            end="",
                            flush=True,
                        )
                        last_percent = percent
                else:
                    print(
                        f"\r  progress: {_format_bytes(downloaded)}",
                        end="",
                        flush=True,
                    )
        if verbose:
            print(flush=True)
    tmp_path.replace(output_path)
    if verbose:
        print(f"Downloaded {name}.", flush=True)


def resolve_model_path(model_path: Union[str, Path],
                       cache_dir: Union[str, Path, None] = None,
                       verbose: bool = True) -> Path:
    """Resolve a local ONNX path or registered model id to a local file.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Existing local file path, or a key from ``selib.urls.MODEL_URLS``.
    cache_dir : str or pathlib.Path, optional
        Directory for downloaded models. Defaults to ``~/.cache/selib/models``.
    verbose : bool, default=True
        If True, print download source, destination, and progress.

    Returns
    -------
    pathlib.Path
        Local path to the ONNX model.
    """
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    model_id = str(model_path)
    if model_id not in MODEL_URLS:
        keys = ", ".join(sorted(MODEL_URLS))
        raise FileNotFoundError(
            f"Model path does not exist and {model_id!r} is not a known model id. "
            f"Known model ids: {keys}")

    entry = MODEL_URLS[model_id]
    rel_path = _cache_relative_path(str(entry["url"]))
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else get_model_cache_dir()
    output_path = cache_root / rel_path
    expected_sha256 = str(entry.get("sha256", ""))

    if output_path.exists():
        if not expected_sha256 or sha256_file(output_path) == expected_sha256:
            return output_path
        output_path.unlink()

    download_file(
        _model_url(str(entry["url"])),
        output_path,
        verbose=verbose,
        label=model_id,
    )
    if expected_sha256:
        actual_sha256 = sha256_file(output_path)
        if actual_sha256 != expected_sha256:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA256 mismatch for downloaded model {model_id!r}: "
                f"expected {expected_sha256}, got {actual_sha256}")
    return output_path
