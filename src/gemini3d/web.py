from __future__ import annotations
import hashlib
import urllib.request
import urllib.error
import socket
import json
import zipfile
import tarfile
import typing as T
from pathlib import Path
import importlib.resources
import shutil
import subprocess

Pathlike = T.Union[str, Path]


def git_download(path: Path, repo: str, tag: str = None):
    """
    Use Git to download code repo.
    """

    git = shutil.which("git")

    if not git:
        raise FileNotFoundError("Git not found.")

    if not tag:
        if not path.is_dir():
            subprocess.check_call([git, "clone", repo, "--depth", "1", str(path)])
        return

    if path.is_dir():
        # don't use "git -C" for old HPC
        ret = subprocess.run([git, "checkout", tag], cwd=str(path))
        if ret.returncode != 0:
            ret = subprocess.run([git, "fetch"], cwd=str(path))
            if ret.returncode != 0:
                raise RuntimeError(f"could not fetch {path}  Maybe try removing this directory.")
            subprocess.check_call([git, "checkout", tag], cwd=str(path))
    else:
        # shallow clone
        subprocess.check_call([git, "clone", repo, "--branch", tag, "--single-branch", str(path)])


def download_and_extract(test_name: str, data_dir: Path) -> Path:

    with importlib.resources.path("gemini3d.tests", "gemini3d_url.json") as url_ini:
        z = get_test_params(test_name, url_ini, data_dir)

        if z["dir"].is_dir():
            return z["dir"]

        try:
            url_retrieve(z["url"], z["zip"], ("md5", z["md5"]))
        except (ConnectionError, ValueError) as e:
            raise ConnectionError(f"problem downloading reference data {e}")

        try:
            extract_zip(z["zip"], z["dir"])
        except zipfile.BadZipFile:
            # bad download, delete and try again (maybe someone hit Ctrl-C during download)
            z["zip"].unlink()
            url_retrieve(z["url"], z["zip"], ("md5", z["md5"]))
            extract_zip(z["zip"], z["dir"])

    return z["dir"]


def get_test_params(test_name: str, url_file: Path, ref_dir: Path) -> dict[str, T.Any]:
    """ get URL and MD5 for a test name """
    json_str = Path(url_file).expanduser().read_text()
    urls = json.loads(json_str)

    z = {
        "url": urls[test_name]["url"],
        "dir": ref_dir / f"test{test_name}",
        "zip": ref_dir / f"test{test_name}.zip",
    }

    if urls[test_name].get("md5"):
        z["md5"] = urls[test_name]["md5"]
    else:
        z["md5"] = None

    return z


def url_retrieve(
    url: str,
    outfile: Pathlike,
    filehash: tuple[str, str] | list[str] = None,
    overwrite: bool = False,
):
    """
    Parameters
    ----------
    url: str
        URL to download from
    outfile: pathlib.Path
        output filepath (including name)
    filehash: tuple of str, str
        hash type (md5, sha1, etc.) and hash
    overwrite: bool
        overwrite if file exists
    """
    outfile = Path(outfile).expanduser().resolve()
    if outfile.is_dir():
        raise ValueError("Please specify full filepath, including filename")
    # need .resolve() in case intermediate relative dir doesn't exist
    if overwrite or not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"{url} => {outfile}")
        try:
            urllib.request.urlretrieve(url, str(outfile))
        except (socket.gaierror, urllib.error.URLError) as err:
            raise ConnectionError(f"could not download {url} due to {err}")

    if filehash and filehash[1]:
        if not file_checksum(outfile, filehash[0], filehash[1]):
            raise ValueError(f"Hash mismatch: {outfile}")


def file_checksum(fn: Path, mode: str, filehash: str) -> bool:
    h = hashlib.new(mode)
    h.update(fn.read_bytes())
    return h.hexdigest() == filehash


def extract_zip(fn: Pathlike, outpath: Pathlike, overwrite: bool = False):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist
    if outpath.is_dir() and not overwrite:
        return

    fn = Path(fn).expanduser().resolve()
    with zipfile.ZipFile(fn) as z:
        z.extractall(str(outpath.parent))


def extract_tar(fn: Pathlike, outpath: Pathlike, overwrite: bool = False):
    outpath = Path(outpath).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist
    if outpath.is_dir() and not overwrite:
        return

    fn = Path(fn).expanduser().resolve()
    if not fn.is_file():
        # tarfile gives confusing error on missing file
        raise FileNotFoundError(fn)

    try:
        with tarfile.open(fn) as z:
            z.extractall(str(outpath.parent))
    except tarfile.TarError as e:
        raise RuntimeError(
            f"""failed to extract {fn} with error {e}.
This file may be corrupt or system libz may be broken.
Try deleting {fn} or manually extracting it."""
        )
