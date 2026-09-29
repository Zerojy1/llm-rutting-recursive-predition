from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE_INDEX = ROOT / "teacher_handoff_20260829"
DESTINATION = WORKSPACE / "老师交付_训练验证测试数据与源代码_20260829"


def decode_name(name: str) -> str:
    return re.sub(
        r"#U([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        name,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source))


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Delivery folder already exists: {DESTINATION}")
    DESTINATION.mkdir(parents=True)

    # 00: human-readable guide and split/code manifests.
    guide_dir = DESTINATION / "00_先读我"
    for name in [
        "README_老师交付说明.md",
        "audit_summary.json",
        "generator_split_summary.csv",
        "downstream_split_summary.csv",
        "main_code_index.csv",
        "all_data_manifest.csv",
        "archive_sha256.txt",
    ]:
        copy_file(SOURCE_INDEX / name, guide_dir / name)

    # 01: original workbooks and canonical processed master table.
    copy_tree(ROOT / "data/raw", DESTINATION / "01_原始数据与母表" / "raw")
    copy_tree(ROOT / "data/processed", DESTINATION / "01_原始数据与母表" / "processed")

    # 02: Qwen chronological data, including audit window tables.
    copy_tree(ROOT / "data/v31_qwen", DESTINATION / "02_Qwen训练验证重拟合与2019保留集")

    # 03: exact downstream arrays exported using the formal split code.
    copy_tree(SOURCE_INDEX / "derived_splits", DESTINATION / "03_TDR_GRU训练验证测试集")

    # 04: all generator/fraction/allocation update pools used by the main experiments.
    copy_tree(ROOT / "data/v31_updates", DESTINATION / "04_全部合成更新数据")

    # 05: all seven structural-OOD fold datasets and fold-specific Q50 updates.
    copy_tree(ROOT / "data/v43_losgo", DESTINATION / "05_LOSGO七折结构泛化数据")

    # 06: top-level human-run entry points, renamed only in this copy for readability.
    entry_dir = DESTINATION / "06_主代码入口"
    for source in sorted(WORKSPACE.glob("*.py")):
        copy_file(source, entry_dir / decode_name(source.name))

    # 07: implementation modules and original TDR-PRSN source.
    copy_tree(ROOT / "scripts", DESTINATION / "07_核心源代码" / "project_scripts")
    copy_tree(ROOT / "first_model", DESTINATION / "07_核心源代码" / "first_model_TDR_PRSN")

    # 08: frozen configuration, tests, environment and protocol documents.
    copy_tree(ROOT / "config", DESTINATION / "08_环境协议与测试" / "config")
    copy_tree(ROOT / "tests", DESTINATION / "08_环境协议与测试" / "tests")
    for name in ["environment_qwen_tdr_v31.yml", "requirements_v31.txt"]:
        copy_file(WORKSPACE / name, DESTINATION / "08_环境协议与测试" / name)
    for pattern in [
        "README_V3.1*.md",
        "V3.1_Final_Experimental_Protocol*.md",
        "V4.0_FROZEN_PROTOCOL*.md",
        "V4.1_STAGE11_FROZEN_PROTOCOL*.md",
        "V4.2_STAGE12_POSTHOC_PROTOCOL*.md",
        "V4.3_STAGE13_STRUCTURAL_OOD_PROTOCOL*.md",
    ]:
        for source in WORKSPACE.glob(pattern):
            copy_file(source, DESTINATION / "08_环境协议与测试" / source.name)

    # Root convenience copy: the teacher should see the guide immediately.
    copy_file(
        SOURCE_INDEX / "README_老师交付说明.md",
        DESTINATION / "README_先读我_数据与代码说明.md",
    )

    rows = []
    for path in sorted(DESTINATION.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(DESTINATION).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = DESTINATION / "00_先读我" / "DELIVERY_FILE_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(row["bytes"] for row in rows)
    print(f"destination={DESTINATION}")
    print(f"files={len(rows) + 1}")
    print(f"bytes={total_bytes + manifest.stat().st_size}")


if __name__ == "__main__":
    main()
