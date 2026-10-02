#!/usr/bin/env python3
"""List or fetch selected members from a remote ZIP using HTTP ranges.

This avoids downloading a multi-gigabyte archive when only a few checkpoints
are required. It supports ordinary (non-ZIP64) archives with stored or deflated
members.
"""

from __future__ import annotations

import argparse
import binascii
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


@dataclass(frozen=True)
class Member:
    name: str
    compression: int
    crc32: int
    compressed_size: int
    size: int
    local_offset: int


def request(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": "myrtavision/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def content_length(url: str) -> int:
    with request(url) as response:
        return int(response.headers["Content-Length"])


def read_range(url: str, start: int, end: int) -> bytes:
    with request(url, start, end) as response:
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"range returned {len(data)} bytes, expected {expected}")
    return data


def members(url: str) -> list[Member]:
    total = content_length(url)
    tail_size = min(total, 65_557)
    tail = read_range(url, total - tail_size, total - 1)
    eocd_at = tail.rfind(EOCD)
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")

    _, disk, central_disk, disk_entries, all_entries, central_size, central_at, _ = struct.unpack_from(
        "<4s4H2IH", tail, eocd_at
    )
    if disk or central_disk or disk_entries != all_entries:
        raise RuntimeError("multi-disk ZIP archives are not supported")
    if central_size == 0xFFFFFFFF or central_at == 0xFFFFFFFF:
        raise RuntimeError("ZIP64 archives are not supported")

    directory = read_range(url, central_at, central_at + central_size - 1)
    found: list[Member] = []
    offset = 0
    for _ in range(all_entries):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, offset)
        if fields[0] != CENTRAL:
            raise RuntimeError(f"invalid central-directory entry at byte {offset}")
        (
            _, _, _, _, compression, _, _, crc32, compressed_size, size,
            name_len, extra_len, comment_len, _, _, _, local_offset,
        ) = fields
        name_start = offset + 46
        name = directory[name_start:name_start + name_len].decode("utf-8")
        found.append(Member(name, compression, crc32, compressed_size, size, local_offset))
        offset = name_start + name_len + extra_len + comment_len
    return found


def fetch(url: str, member: Member, destination: Path) -> None:
    header = read_range(url, member.local_offset, member.local_offset + 29)
    signature, _, _, compression, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<4s5H3I2H", header
    )
    if signature != LOCAL or compression != member.compression:
        raise RuntimeError(f"invalid local header for {member.name}")
    data_at = member.local_offset + 30 + name_len + extra_len

    destination.parent.mkdir(parents=True, exist_ok=True)
    crc = 0
    written = 0
    decompressor = zlib.decompressobj(-15) if member.compression == 8 else None
    if member.compression not in (0, 8):
        raise RuntimeError(f"unsupported compression method {member.compression}")

    with request(url, data_at, data_at + member.compressed_size - 1) as response, destination.open("wb") as out:
        remaining = member.compressed_size
        while remaining:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"truncated download for {member.name}")
            remaining -= len(chunk)
            decoded = decompressor.decompress(chunk) if decompressor else chunk
            out.write(decoded)
            written += len(decoded)
            crc = binascii.crc32(decoded, crc)
        if decompressor:
            decoded = decompressor.flush()
            out.write(decoded)
            written += len(decoded)
            crc = binascii.crc32(decoded, crc)

    if written != member.size or crc & 0xFFFFFFFF != member.crc32:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"integrity check failed for {member.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("member", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()

    available = members(args.url)
    if args.list:
        for item in available:
            print(f"{item.size:12d} {item.compressed_size:12d} {item.name}")
        return

    by_name = {item.name: item for item in available}
    for name in args.member:
        item = by_name.get(name)
        if item is None:
            raise SystemExit(f"member not found: {name}")
        destination = args.output / Path(name).name
        print(f"fetching {name} -> {destination} ({item.size / 1024**2:.1f} MiB)")
        fetch(args.url, item, destination)


if __name__ == "__main__":
    main()
