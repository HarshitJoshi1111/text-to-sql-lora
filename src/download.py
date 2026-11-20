"""Stage 1: download WikiSQL (questions, logical forms and tables) from the authors' GitHub repository.

Only the JSON files are extracted (62 MB); the SQLite files in the archive are not needed, because
every table is rebuilt in memory from its JSON rows (src/wikisql.py).
"""
import io
import tarfile

import requests

from src.params import RAW, load_params

FILES = [f"{split}{suffix}" for split in ("train", "dev", "test") for suffix in (".jsonl", ".tables.jsonl")]


def main() -> None:
    url = load_params()["data"]["url"]
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:bz2") as archive:
        for member in archive.getmembers():
            name = member.name.split("/")[-1]
            if member.isfile() and name in FILES:
                (RAW / name).write_bytes(archive.extractfile(member).read())
                print(f"{name}: {member.size / 1e6:.1f} MB")
    missing = [f for f in FILES if not (RAW / f).exists()]
    if missing:
        raise RuntimeError(f"missing in the archive: {missing}")


if __name__ == "__main__":
    main()
