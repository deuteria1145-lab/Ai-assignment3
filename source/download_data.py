"""Download and verify the official Fashion-MNIST IDX archives."""

import argparse
import gzip
import hashlib
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
ARCHIVES = {
    "train-images-idx3-ubyte.gz": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", (2051, 60000, 28, 28)),
    "train-labels-idx1-ubyte.gz": ("25c81989df183df01b3e8a0aad5dffbe", (2049, 60000)),
    "t10k-images-idx3-ubyte.gz": ("bef4ecab320f06d8554ea6380940ec79", (2051, 10000, 28, 28)),
    "t10k-labels-idx1-ubyte.gz": ("bb300cfdad3c16e7a12a480ee83cd310", (2049, 10000)),
}


def verify_archive(compressed, filename, expected_md5, expected_header):
    """Check the official checksum, IDX metadata, payload length, and label counts."""
    digest = hashlib.md5(compressed, usedforsecurity=False).hexdigest()
    if digest != expected_md5:
        raise ValueError(f"{filename} has checksum {digest}; expected {expected_md5}")

    content = gzip.decompress(compressed)
    layout = struct.Struct(">" + "I" * len(expected_header))
    if layout.unpack_from(content) != expected_header:
        raise ValueError(f"{filename} has an unexpected IDX header")

    sample_count = expected_header[1]
    if len(expected_header) == 4:
        expected_size = sample_count * expected_header[2] * expected_header[3]
    else:
        expected_size = sample_count
        labels = content[layout.size:]
        if any(labels.count(label) != sample_count // 10 for label in range(10)):
            raise ValueError(f"{filename} has unexpected label counts")

    if len(content) - layout.size != expected_size:
        raise ValueError(f"{filename} has an unexpected payload length")


def ensure_archive(target, expected_md5, expected_header):
    """Keep a verified local archive or atomically replace it with a verified download."""
    if target.is_file():
        try:
            verify_archive(target.read_bytes(), target.name, expected_md5, expected_header)
        except ValueError as error:
            print(f"{error}. Downloading a verified copy.")
        else:
            print(f"Verified existing {target.name}")
            return

    with urlopen(BASE_URL + target.name, timeout=60) as response:
        compressed = response.read()
    verify_archive(compressed, target.name, expected_md5, expected_header)

    with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
        part = Path(stream.name)
        try:
            stream.write(compressed)
            stream.close()
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    print(f"Downloaded and verified {target.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_data = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument("--data-dir", default=workspace_data, type=Path)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, (expected_md5, expected_header) in ARCHIVES.items():
        ensure_archive(args.data_dir / name, expected_md5, expected_header)


if __name__ == "__main__":
    main()
