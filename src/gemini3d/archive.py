from __future__ import annotations
import zipfile
import tarfile
import typing as T
from pathlib import Path
import tempfile

try:
    import zstandard
except ImportError:
    zstandard = None  # type: ignore

Pathlike = T.Union[str, Path]


def extract_zst(archive: Pathlike, out_path: Pathlike):
    """extract .zst file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------

    archive: pathlib.Path or str
      .zst file to extract

    out_path: pathlib.Path or str
      directory to extract files and directories to
    """
    if zstandard is None:
        raise ImportError("pip install zstandard")

    archive = Path(archive).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            dctx.copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)


def extract_zip(archive: Pathlike, outpath: Pathlike):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

    archive = Path(archive).expanduser().resolve()
    with zipfile.ZipFile(archive) as z:
        z.extractall(outpath)


def extract_tar(archive: Pathlike, outpath: Pathlike):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

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
