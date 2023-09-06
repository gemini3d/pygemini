from __future__ import annotations
import urllib.request
import urllib.error
import hashlib
import socket
import json
import typing as T
from pathlib import Path
import importlib.resources
import shutil
import subprocess

from .archive import extract


def git_download(path: Path, repo: str, tag: str | None = None):
    """
    Use Git to download code repo.
    """

    git = shutil.which("git")

    if not git:
        raise FileNotFoundError("Git not found.")

    if not tag:
        if not path.is_dir():
            subprocess.check_call([git, "clone", repo, str(path)])
        return

    if path.is_dir():
        if not (path / ".git").is_dir():
            raise EnvironmentError(
                f"{path} exists but is not a Git repo. Try specifying a local Git repo or new (uncreated) directory."
            )
        # don't use "git -C" for old HPC
        ret = subprocess.run([git, "checkout", tag], cwd=path)
        if ret.returncode != 0:
            ret = subprocess.run([git, "fetch"], cwd=path)
            if ret.returncode != 0:
                raise RuntimeError(
                    f"could not Git fetch {path}  Maybe try removing this directory."
                )
            subprocess.check_call([git, "checkout", tag], cwd=path)
    else:
        subprocess.check_call([git, "clone", repo, path])
        subprocess.check_call([git, "checkout", tag], cwd=path)


def download_and_extract(test_name: str, data_dir: Path) -> Path:
    ref_file = data_dir / "ref_data.json"

    if not ref_file.is_file():
        jmeta = json.loads(
            importlib.resources.files("gemini3d").joinpath("libraries.json").read_text()
        )
        url_retrieve(url=jmeta["ref_data"]["url"], outfile=ref_file)

    z = get_test_params(test_name, url_file=ref_file, ref_dir=data_dir)

    if z["dir"].is_dir():
        return z["dir"]

    url_retrieve(z["url"], z["archive"], ("sha256", z["sha256"]))

    extract(z["archive"], z["dir"])

    return z["dir"]


def get_test_params(test_name: str, url_file: Path, ref_dir: Path) -> dict[str, T.Any]:
    """get URL and hash for a test name"""

    urls = json.loads(Path(url_file).expanduser().read_text())

    tests = urls["tests"][test_name]

    z = {
        "url": tests["url"],
        "dir": ref_dir / test_name,
        "archive": ref_dir / tests["archive"],
    }

    if tests.get("sha256"):
        z["sha256"] = tests["sha256"]
    else:
        z["sha256"] = None

    return z


def url_retrieve(
    url: str,
    outfile: Path,
    filehash: tuple[str, str] | None = None,
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
        hash type (md5, sha256, etc.) and hash
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
            raise ValueError(f"Hash mismatch: {outfile} from {url}")


def file_checksum(fn: Path, mode: str, filehash: str) -> bool:
    h = hashlib.new(mode)
    h.update(fn.read_bytes())
    return h.hexdigest() == filehash
