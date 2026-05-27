#!/usr/bin/env python3
"""Normalize custom-instruction marker blocks and populate file-specific content.

Markers are always:
  BEGIN CUSTOM INSTRUCTION MARKER
  END CUSTOM INSTRUCTION MARKER

For each config entry in config/local_anchors.json, this tool will:
1) Find the configured anchor text.
2) If a marker block already exists adjacent to the anchor, collapse it to an
   empty block (remove any content inside).
3) If no marker block exists there, insert a new empty marker block within one
   blank line of the anchor.
4) For files with a registered content generator, fill the marker block with
    generated lines from config/instruction_registry.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import shutil


MARKER_TEXT_START = "BEGIN CUSTOM INSTRUCTION MARKER"
MARKER_TEXT_END = "END CUSTOM INSTRUCTION MARKER"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "local_anchors.json"
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "config" / "instruction_registry.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_instr_dict_path() -> Path:
    candidates = [
        REPO_ROOT / "custom-instructions" / "riscv-opcodes" / "instr_dict.json",
        REPO_ROOT / "riscv-opcodes" / "instr_dict.json",
        REPO_ROOT / "riscv-gnu-toolchain" / "riscv-opcodes" / "instr_dict.json",
        REPO_ROOT.parent / "riscv-workspace" / "riscv-gnu-toolchain" / "riscv-opcodes" / "instr_dict.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not locate instr_dict.json in any known workspace layout")


@dataclass
class InstructionRecord:
    name: str
    status: str
    format: str
    opcode: str
    funct3: int | None = None
    funct7: int | None = None
    match: str | None = None
    mask: str | None = None


def detect_newline(text: str) -> str:
    # Inspect only the first chunk for performance on large files
    head = text[:4096]
    if "\r\n" in head:
        return "\r\n"
    return "\n"


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_target_path(raw_path: str, repo_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p

    # Prefer local custom-instructions checkout under the workspace so we don't
    # accidentally edit files in sibling workspaces. Then check common
    # toolchain locations.
    candidate_roots = [
        repo_root / "custom-instructions",
        repo_root,
        repo_root / "riscv-gnu-toolchain",
        repo_root.parent / "riscv-workspace" / "riscv-gnu-toolchain",
    ]
    for root in candidate_roots:
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate

    return (repo_root / p).resolve()


def get_anchor_spec(entry: Dict) -> Tuple[str, str]:
    if "anchor" in entry and isinstance(entry["anchor"], dict):
        match = entry["anchor"].get("match")
        placement = entry["anchor"].get("placement", "after")
    else:
        match = entry.get("local_anchor")
        placement = entry.get("placement", "after")

    if not match or not isinstance(match, str):
        raise ValueError("entry is missing anchor match text")
    if placement not in ("before", "after"):
        raise ValueError(f"unsupported placement: {placement}")
    return match, placement


def make_commented_markers(path: Path) -> Tuple[str, str]:
    """Return start and end marker strings wrapped in appropriate comments.

    For C headers/sources use /* ... */. For other files use # ...
    """
    ext = Path(path).suffix.lower()
    c_style = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}
    if ext in c_style:
        start = f"/* {MARKER_TEXT_START} */"
        end = f"/* {MARKER_TEXT_END} */"
    else:
        # Default to hash-style comment (shell/Makefile/assembly)
        start = f"# {MARKER_TEXT_START}"
        end = f"# {MARKER_TEXT_END}"
    return start, end


def normalize_block_around_anchor_with_markers(
    text: str, anchor: str, placement: str, start_marker: str, end_marker: str, body: str = ""
) -> Tuple[str, bool, str]:
    """Normalize marker block using provided comment-wrapped markers.

    Returns (new_text, changed, action)
    """
    nl = detect_newline(text)
    body = body.rstrip()
    if body:
        empty_block = f"{start_marker}{nl}{body}{nl}{end_marker}"
    else:
        empty_block = f"{start_marker}{nl}{end_marker}"

    anchor_idx = text.find(anchor)
    if anchor_idx < 0:
        raise ValueError("anchor text not found in target file")

    anchor_start = anchor_idx
    anchor_end = anchor_idx + len(anchor)

    if placement == "after":
        # Anchor + up to one blank line + marker block
        pattern = re.compile(
            re.escape(anchor)
            + r"(?:\r?\n){1,2}"
            + re.escape(start_marker)
            + r"(?:\r?\n).*?"
            + re.escape(end_marker)
            + r"(?:\r?\n)?",
            re.DOTALL,
        )
        m = pattern.search(text)
        replacement = f"{anchor}{nl}{nl}{empty_block}{nl}"
        if m:
            out = text[: m.start()] + replacement + text[m.end() :]
            return out, out != text, "cleared"

        insert = f"{nl}{nl}{empty_block}"
        out = text[:anchor_end] + insert + text[anchor_end:]
        return out, True, "inserted"

    # placement == "before"
    pattern = re.compile(
        re.escape(start_marker)
        + r"(?:\r?\n).*?"
        + re.escape(end_marker)
        + r"(?:\r?\n){1,2}"
        + re.escape(anchor),
        re.DOTALL,
    )
    m = pattern.search(text)
    replacement = f"{empty_block}{nl}{nl}{anchor}"
    if m:
        out = text[: m.start()] + replacement + text[m.end() :]
        return out, out != text, "cleared"

    insert = f"{empty_block}{nl}{nl}"
    out = text[:anchor_start] + insert + text[anchor_start:]
    return out, True, "inserted"


def normalize_riscv_insn_list_custom_entry(
    text: str, start_marker: str, end_marker: str
) -> Tuple[str, bool, str]:
    """Ensure $(riscv_insn_ext_custom) is present in riscv_insn_list.

    This path intentionally does not place marker comments inside the
    continuation list because comment lines can break Makefile parsing.
    """
    nl = detect_newline(text)
    custom_line = "\t$(riscv_insn_ext_custom) \\"

    list_head = re.search(r"^riscv_insn_list\s*=\s*\\\s*$", text, re.MULTILINE)
    if not list_head:
        raise ValueError("riscv_insn_list assignment not found")

    block_start = list_head.start()
    block_after_head = list_head.end()

    # The list block ends at the first blank line after the assignment.
    tail = text[block_after_head:]
    blank = re.search(r"(?:\r?\n){2}", tail)
    if blank:
        block_end = block_after_head + blank.start()
    else:
        block_end = len(text)

    list_block = text[block_start:block_end]

    # Remove legacy marker lines if they were previously inserted.
    cleaned = re.sub(rf"^.*{re.escape(start_marker)}.*(?:\r?\n)?", "", list_block, flags=re.MULTILINE)
    cleaned = re.sub(rf"^.*{re.escape(end_marker)}.*(?:\r?\n)?", "", cleaned, flags=re.MULTILINE)

    # Ensure exactly one custom extension entry line in the list block.
    line_pat = re.compile(r"^\s*\$\(riscv_insn_ext_custom\)\s*\\\s*$", re.MULTILINE)
    if not line_pat.search(cleaned):
        prefix = "" if cleaned.endswith(("\n", "\r")) else nl
        cleaned = f"{cleaned}{prefix}{custom_line}{nl}"

    out = text[:block_start] + cleaned + text[block_end:]
    if out == text:
        return out, False, "unchanged"
    return out, True, "updated"


def build_instruction_catalog(registry: Dict, instr_dict: Dict) -> List[InstructionRecord]:
    # Compatibility wrapper: build from registry then enrich from instr_dict
    records = build_catalog_from_registry(registry)
    names = {r.name: r for r in records}
    for name, entry in instr_dict.items():
        extensions = entry.get("extension") or []
        if "rv_xcustom" not in extensions:
            continue
        # riscv-opcodes may emit instruction names prefixed with 'custom_'.
        # Accept either the raw name or a 'custom_'-prefixed variant when
        # enriching records from instr_dict.
        candidate = name
        if name not in names and name.startswith("custom_"):
            candidate = name[len("custom_"):]
        if candidate not in names:
            continue
        match = entry.get("match")
        mask = entry.get("mask")
        if not match or not mask:
            raise ValueError(f"{name}: missing match or mask")
        rec = names[name]
        rec.match = str(match)
        rec.mask = str(mask)
    return records


def build_catalog_from_registry(registry: Dict) -> List[InstructionRecord]:
    records: List[InstructionRecord] = []
    for entry in registry.get("entries", []):
        name = str(entry.get("name", ""))
        encoding = entry.get("encoding") or {}
        record = InstructionRecord(
            name=name,
            status=str(entry.get("status", "")),
            format=str(entry.get("format", "")),
            opcode=str(entry.get("opcode", "")),
            funct3=encoding.get("funct3"),
            funct7=encoding.get("funct7"),
        )
        records.append(record)
    return records


def find_riscv_opcodes_dir() -> Path:
    candidates = [
        REPO_ROOT / "custom-instructions" / "riscv-opcodes",
        REPO_ROOT / "riscv-opcodes",
        REPO_ROOT / "riscv-gnu-toolchain" / "riscv-opcodes",
        REPO_ROOT.parent / "riscv-workspace" / "riscv-gnu-toolchain" / "riscv-opcodes",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError("could not locate riscv-opcodes directory")


def enrich_catalog_from_instr_dict(records: List[InstructionRecord]) -> None:
    """Run riscv-opcodes build and update `records` in-place with match/mask."""
    opdir = find_riscv_opcodes_dir()
    res = subprocess.run(["make"], cwd=str(opdir))
    if res.returncode != 0:
        raise SystemExit("riscv-opcodes build failed")
    instr_path = resolve_instr_dict_path()
    instr_dict = load_json(instr_path)
    names = {r.name: r for r in records}
    for name, entry in instr_dict.items():
        exts = entry.get("extension") or []
        if "rv_xcustom" not in exts:
            continue
        candidate = name
        if name not in names and name.startswith("custom_"):
            candidate = name[len("custom_"):]
        if candidate not in names:
            continue
        match = entry.get("match")
        mask = entry.get("mask")
        if not match or not mask:
            raise ValueError(f"{name}: missing match or mask")
        rec = names[candidate]
        rec.match = str(match)
        rec.mask = str(mask)


def render_rv_xcustom_body(records: List[InstructionRecord]) -> str:
    lines = []
    for record in records:
        if record.status not in ("allocated", "applied"):
            continue
        if record.format != "R":
            continue

        try:
            opcode_bits = int(str(record.opcode), 0) >> 2
            funct3 = int(record.funct3)
            funct7 = int(record.funct7)
        except Exception as exc:
            raise ValueError(f"{record.name}: invalid R-type registry entry") from exc

        base = record.name.replace('.', '_')
        pref = f"custom_{base}"
        lines.append(
            f"{pref} rd rs1 rs2 31..25={funct7} 14..12={funct3} 6..2=0x{opcode_bits:02x} 1..0=3"
        )

    return "\n".join(lines)


def rendor_match_mask_body(records: List[InstructionRecord]) -> str:
    """Render MATCH/MASK #defines for instructions in the rv_xcustom extension."""
    lines = []
    for record in records:
        if record.match is None or record.mask is None:
            continue
        symbol = record.name.upper().replace(".", "_")
        lines.append(f"#define MATCH_{symbol} {record.match}")
        lines.append(f"#define MASK_{symbol} {record.mask}")

    return "\n".join(lines)


def render_riscv_opc_c_body(records: List[InstructionRecord]) -> str:
    lines = []
    for record in records:
        if record.status not in ("allocated", "applied"):
            continue
        if record.match is None or record.mask is None:
            continue
        if record.format != "R":
            continue

        symbol = record.name.upper().replace(".", "_")
        base = record.name.replace('.', '_')
        pref = f"custom_{base}"
        lines.append(
            f'{{"{pref}",    0, INSN_CLASS_I, "d,s,t", MATCH_{symbol}, MASK_{symbol}, match_opcode, 0 }},'
        )

    return "\n".join(lines)


def render_declare_insn_body(records: List[InstructionRecord]) -> str:
    lines = []
    for record in records:
        if record.status not in ("allocated", "applied"):
            continue
        if record.match is None or record.mask is None:
            continue
        symbol = record.name.upper().replace(".", "_")
        base = record.name.replace('.', '_')
        pref = f"custom_{base}"
        lines.append(f"DECLARE_INSN({pref}, MATCH_{symbol}, MASK_{symbol})")
    return "\n".join(lines)


def render_file_body(entry: Dict, file_name: str, records: List[InstructionRecord]) -> str:
    # Select renderer based on file target and anchor context in `entry`.
    fname = file_name
    if fname == "rv_xcustom":
        return render_rv_xcustom_body(records)
    if fname == "riscv-opc.c":
        return render_riscv_opc_c_body(records)
    if fname == "riscv.mk.in":
        # Construct a makefile variable listing for custom instructions.
        lines = []
        lines.append("riscv_insn_ext_custom = \\")
        for record in records:
            if record.status not in ("allocated", "applied"):
                continue
            base = record.name.replace('.', '_')
            pref = f"custom_{base}"
            lines.append(f"\t{pref} \\")
        return "\n".join(lines)
    if fname in ("riscv-opc.h", "encoding.h"):
        anchor = entry.get("anchor") or {}
        match_text = anchor.get("match", "") if isinstance(anchor, dict) else ""
        if "DECLARE_INSN" in match_text:
            return render_declare_insn_body(records)
        return rendor_match_mask_body(records)
    return ""


def process_entry(
    entry: Dict, repo_root: Path, records: List[InstructionRecord], dry_run: bool = False
) -> Tuple[bool, str, str]:
    file_name = str(entry.get("file_name", "<unnamed>"))
    raw_path = entry.get("path")
    if not raw_path or not isinstance(raw_path, str):
        raise ValueError(f"{file_name}: missing path")

    anchor_match, placement = get_anchor_spec(entry)
    target_path = resolve_target_path(raw_path, repo_root)

    if not target_path.exists():
        raise FileNotFoundError(f"{file_name}: target does not exist: {target_path}")

    text = target_path.read_text(encoding="utf-8")
    start_marker, end_marker = make_commented_markers(target_path)
    body = render_file_body(entry, target_path.name, records)
    if target_path.name == "riscv.mk.in" and placement == "after" and "riscv_insn_list" in anchor_match:
        updated, changed, action = normalize_riscv_insn_list_custom_entry(
            text, start_marker, end_marker
        )
    else:
        updated, changed, action = normalize_block_around_anchor_with_markers(
            text, anchor_match, placement, start_marker, end_marker, body=body
        )

    if changed and not dry_run:
        target_path.write_text(updated, encoding="utf-8")

    return changed, action, str(target_path)


def manage_spike_behavior_files(repo_root: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Remove existing Spike behavior files starting with 'custom_' and copy
    all files from `custom-instructions/behavior_files` into Spike's insns
    directory, prefixing each copied filename with 'custom_'.

    Returns (removed_count, copied_count).
    """
    dst_dir = repo_root / "riscv-gnu-toolchain" / "spike" / "riscv" / "insns"

    if not dst_dir.exists() or not dst_dir.is_dir():
        raise FileNotFoundError(f"spike insns directory not found: {dst_dir}")

    # Load registry and select behavior files for active instructions only
    registry_path = DEFAULT_REGISTRY
    try:
        registry = load_json(registry_path)
    except FileNotFoundError:
        return 0, 0

    entries = registry.get("entries", []) or []
    to_copy: List[Path] = []
    for entry in entries:
        status = entry.get("status", "")
        if status not in ("allocated", "applied"):
            continue
        bf = entry.get("behavior_file")
        if not bf:
            continue
        src = (repo_root / "custom-instructions" / bf).resolve()
        if src.exists() and src.is_file():
            to_copy.append(src)

    removed = 0
    for p in list(dst_dir.iterdir()):
        if p.is_file() and p.name.startswith("custom_"):
            if dry_run:
                print(f"[behaviors] would remove: {p}")
            else:
                p.unlink()
            removed += 1

    copied = 0
    for p in sorted(to_copy):
        dest = dst_dir / ("custom_" + p.name)
        if dry_run:
            print(f"[behaviors] would copy: {p} -> {dest}")
        else:
            shutil.copy2(p, dest)
        copied += 1

    return removed, copied


def run(config_path: Path, dry_run: bool = False) -> int:
    cfg = load_config(config_path)
    registry = load_json(DEFAULT_REGISTRY)
    if not isinstance(registry, dict):
        raise SystemExit(f"missing or invalid registry file: {DEFAULT_REGISTRY}")
    # Phase A: build catalog from registry and populate rv_xcustom
    records = build_catalog_from_registry(registry)
    entries = cfg.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("config must contain an 'entries' list")

    changed_count = 0

    # Phase A: render and insert rv_xcustom (or any entries intended to be
    # generated from the registry before running riscv-opcodes).
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"entries[{idx}] must be an object")
        if str(entry.get("file_name", "")) != "rv_xcustom":
            continue
        try:
            changed, action, path_str = process_entry(entry, REPO_ROOT, records, dry_run=dry_run)
            changed_count += int(changed)
            print(f"[phaseA:{idx}] {action}: {path_str}")
        except Exception as exc:
            raise SystemExit(f"entries[{idx}] failed in phase A: {exc}") from exc

    # Phase A done. Now build riscv-opcodes and enrich catalog with MATCH/MASK.
    if not dry_run:
        enrich_catalog_from_instr_dict(records)
    else:
        # In dry-run, attempt to load existing instr_dict if present to allow
        # subsequent renderers to show expected output without running make.
        try:
            instr_path = resolve_instr_dict_path()
            instr_dict = load_json(instr_path)
            # Update records from instr_dict without running make
            names = {r.name: r for r in records}
            for name, entry in instr_dict.items():
                exts = entry.get("extension") or []
                if "rv_xcustom" not in exts:
                    continue
                if name not in names:
                    continue
                match = entry.get("match")
                mask = entry.get("mask")
                if match and mask:
                    rec = names[name]
                    rec.match = str(match)
                    rec.mask = str(mask)
        except FileNotFoundError:
            # No instr_dict available in dry-run; continue.
            pass

    # Phase B: render remaining entries (riscv-opc.c, encoding.h, etc.)
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"entries[{idx}] must be an object")
        if str(entry.get("file_name", "")) == "rv_xcustom":
            continue
        try:
            changed, action, path_str = process_entry(entry, REPO_ROOT, records, dry_run=dry_run)
            changed_count += int(changed)
            print(f"[phaseB:{idx}] {action}: {path_str}")
        except Exception as exc:
            raise SystemExit(f"entries[{idx}] failed in phase B: {exc}") from exc

    mode = "dry-run" if dry_run else "apply"
    print(f"done ({mode}): {changed_count} file(s) changed")

    # Manage Spike behavior files: remove existing `custom_` prefixed files
    # and copy behavior snippets from our `custom-instructions/behavior_files`
    try:
        removed, copied = manage_spike_behavior_files(REPO_ROOT, dry_run=dry_run)
        print(f"[behaviors] removed:{removed} copied:{copied}")
    except Exception as exc:
        print(f"[behaviors] skipped: {exc}")
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Insert/normalize custom instruction marker blocks around local anchors")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to local_anchors.json")
    p.add_argument("--dry-run", action="store_true", help="show intended changes without writing files")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run(Path(args.config).expanduser().resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
