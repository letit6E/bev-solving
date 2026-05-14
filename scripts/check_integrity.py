"""Verify the unzipped dataset folder against its zip archive.

Quick mode compares file sizes; --crc recomputes CRC32 (slow but exact);
--repair re-extracts only the missing or corrupted files.

Example:
    python scripts/check_integrity.py --zip autonomy_yandex_dataset_train_v2.zip \
        --root autonomy_yandex_dataset_train --crc --repair
"""
import argparse
import sys
import zipfile
import zlib
from pathlib import Path


def crc32_of_file(path, chunk=1 << 20):
    c = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def check_one_zip(zip_path, root, use_crc=False, repair=False):
    missing, corrupted = [], []
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total = sum(1 for i in infos if not i.is_dir())
        for info in infos:
            if info.is_dir():
                continue
            target = root / info.filename
            if not target.exists():
                missing.append(info.filename)
            elif target.stat().st_size != info.file_size:
                corrupted.append(info.filename)
            elif use_crc and crc32_of_file(target) != info.CRC:
                corrupted.append(info.filename)
        if repair and (missing or corrupted):
            bad = sorted(set(missing) | set(corrupted))
            print(f"repairing {len(bad)} files from {zip_path.name}")
            for name in bad:
                try:
                    zf.extract(name, path=root)
                except KeyError:
                    pass
    return missing, corrupted, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", nargs="+", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--crc", action="store_true")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--report", default="integrity_report.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    all_missing, all_corrupted, grand_total = [], [], 0
    for z in args.zip:
        zp = Path(z).resolve()
        if not zp.exists():
            print(f"!! zip not found: {zp}")
            continue
        print(f"\n=== {zp.name} ===")
        m, c, t = check_one_zip(zp, root, args.crc, args.repair)
        all_missing.extend(m)
        all_corrupted.extend(c)
        grand_total += t
        print(f"  entries={t}  missing={len(m)}  corrupted={len(c)}")

    Path(args.report).write_text(
        f"# total={grand_total}  missing={len(all_missing)}  corrupted={len(all_corrupted)}\n"
        "## MISSING\n" + "\n".join(all_missing) + "\n## CORRUPTED\n" + "\n".join(all_corrupted))
    return 1 if (all_missing or all_corrupted) and not args.repair else 0


if __name__ == "__main__":
    sys.exit(main())
