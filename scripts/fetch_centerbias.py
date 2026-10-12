#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch the external DeepGaze center-bias template and verify its checksum."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
SHA256 = "051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9"


def fetch(output):
    output = Path(output)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest() == SHA256:
            return output
        raise FileExistsError(f"Existing file has a different checksum: {output}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Center-bias download checksum does not match the archived template")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("resources/centerbias_mit1003.npy"))
    args = parser.parse_args()
    print(fetch(args.output))
