from __future__ import annotations
import zipfile
import tarfile
import typing as T
from pathlib import Path
import tempfile

try:
    import zstandard
except ImportError:
    zstandard = None

Pathlike = T.Union[str, Path]


def extract_zst(archive: Pathlike, out_path: Pathlike, overwrite: bool = False):
    if zstandard is None:
        raise ImportError("pip install zstandard")

    archive = Path(archive).expanduser().resolve()
    out_path = Path(out_path).expanduser()
    out_path.mkdir(exist_ok=True)

    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            dctx.copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)


def extract_zip(archive: Pathlike, outpath: Pathlike, overwrite: bool = False):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist
    if outpath.is_dir() and not overwrite:
        return

    archive = Path(archive).expanduser().resolve()
    with zipfile.ZipFile(archive) as z:
        z.extractall(outpath)


def extract_tar(archive: Pathlike, outpath: Pathlike, overwrite: bool = False):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist
    if outpath.is_dir() and not overwrite:
        return

    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        # tarfile gives confusing error on missing file
        raise FileNotFoundError(archive)

    try:
        with tarfile.open(archive) as z:
            z.extractall(outpath)
    except tarfile.TarError as e:
        raise RuntimeError(
            f"""failed to extract {archive} with error {e}.
This file may be corrupt or system libz may be broken.
Try deleting {archive} or manually extracting it."""
        )
