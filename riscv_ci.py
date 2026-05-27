#!/usr/bin/env python3
"""riscv_ci.py - simple allocator CLI for custom RISC-V instructions

Commands implemented:
  allocate [--dry-run] [--apply]   Validate manifest, assign opcode/funct fields
    purge [--apply]                  Remove tombstoned/removed entries and owned files
  validate                         Validate `instructions.yaml` only
  show [--name NAME]               Show entries from instruction_registry.json

Notes:
  - This prototype uses `instruction_registry.json` as the authoritative
    registry and performs allocation by appending entries to it on --apply.
  - Requires PyYAML for parsing `instructions.yaml`.
"""
import argparse
import copy
import json
import os
import shutil
import sys
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import yaml
except Exception:
    setup_path = os.path.join(os.path.dirname(__file__), "setup.sh")
    print("Missing dependency: PyYAML.", file=sys.stderr)
    print(f"Run '{setup_path}' to create a virtualenv and install dependencies, or: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(2)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_file(path):
    if not os.path.exists(path):
        return
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = f"{path}.{ts}.bak"
    shutil.copy2(path, bak)
    return bak


def ordered_registry_entry(name, fmt_name, opcode, behavior_file, encoding, status, source, allocated_at):
    return {
        "name": name,
        "format": fmt_name,
        "opcode": opcode,
        "behavior_file": behavior_file,
        "encoding": encoding,
        "status": status,
        "source": source,
        "allocated_at": allocated_at,
    }


def parse_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit("Bad manifest format: expected mapping at root")
    instructions = data.get("instructions")
    if instructions is None:
        data["instructions"] = []
    return data


def validate_manifest(manifest, format_registry):
    errors = []
    warnings = []
    instructions = manifest.get("instructions")
    if instructions is None:
        warnings.append("`instructions` list missing in manifest; treating as empty")
        instructions = []
    elif not isinstance(instructions, list):
        errors.append("`instructions` must be a list when present in manifest")
        return errors, warnings
    formats = format_registry.get("formats", {})
    seen_names = set()
    for idx, instr in enumerate(instructions):
        ctx = f"instructions[{idx}]"
        if not isinstance(instr, dict):
            errors.append(f"{ctx}: entry must be a mapping")
            continue
        for req in ("name", "format"):
            if req not in instr:
                errors.append(f"{ctx}: missing required field `{req}`")
        name = instr.get("name")
        if name:
            if name in seen_names:
                errors.append(f"{ctx}: duplicate instruction name `{name}` in manifest")
            seen_names.add(name)
        fmt = instr.get("format")
        if fmt not in formats:
            errors.append(f"{ctx}: unknown format `{fmt}` (allowed: {list(formats.keys())})")
    return errors, warnings


def registry_entry_map(registry):
    entries = {}
    if not registry:
        return entries
    for entry in registry.get("entries", []):
        name = entry.get("name")
        if name:
            entries[name] = entry
    return entries


def manifest_names(manifest):
    names = []
    for instr in manifest.get("instructions") or []:
        name = instr.get("name")
        if name:
            names.append(name)
    return names


def toolchain_root(base_dir, override=None):
    if override:
        return Path(override).expanduser().resolve()
    env_root = os.environ.get("RISCV_GNU_TOOLCHAIN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(base_dir).parent / "riscv-gnu-toolchain"


def riscv_opcodes_root(base_dir, override=None):
    env_root = os.environ.get("RISCV_OPCODES_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if override:
        return Path(override).expanduser().resolve().parent / "riscv-opcodes"
    return Path(base_dir).parent / "riscv-opcodes"


def path_state(path_obj):
    if not path_obj.exists():
        return "missing"
    if path_obj.is_dir() and not any(path_obj.iterdir()):
        return "empty"
    return "present"


def repo_state(repo_root):
    if not repo_root.exists():
        return "missing"
    if not (repo_root / ".git").exists():
        return "not-a-repo"
    return "present"


def status_icon(state):
    if state == "present":
        return "✅"
    if state == "missing":
        return "❌"
    if state in ("empty", "not-a-repo"):
        return "⚠️"
    return "ℹ️"


def repo_link(repo_name):
    return f"https://github.com/{repo_name}"


RV_XCUSTOM_HEADER = "# RISC-V Custom Instructions (auto-generated)\n# DO NOT EDIT - managed by add_custom_instructions.py\n\n"


def parse_gitmodules(repo_root):
    gitmodules = repo_root / ".gitmodules"
    modules = {}
    if not gitmodules.exists():
        return modules

    current_path = None
    current_url = None
    with open(gitmodules, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[submodule "):
                if current_path:
                    modules[current_path] = current_url
                current_path = None
                current_url = None
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if key == "path":
                current_path = value
            elif key == "url":
                current_url = value

    if current_path:
        modules[current_path] = current_url
    return modules


def doctor_riscv_opcodes(base_dir, fix=False, assume_yes=False, toolchain_root_override=None):
    target_root = toolchain_root(base_dir, override=toolchain_root_override)
    opcodes_root = riscv_opcodes_root(base_dir, override=toolchain_root_override)
    rv_xcustom_path = opcodes_root / "extensions" / "rv_xcustom"
    rv_xcustom_parent = rv_xcustom_path.parent
    encoding_out_path = opcodes_root / "encoding.out.h"
    instr_dict_path = opcodes_root / "instr_dict.json"
    spike_root = target_root / "spike"
    spike_encoding_h = spike_root / "riscv" / "encoding.h"

    unresolved = []
    max_attempts = 6
    for _ in range(max_attempts):
        results = []
        gitmodules = parse_gitmodules(target_root)
        spike_repo_url = gitmodules.get("spike")
        toolchain_repo_state = repo_state(target_root)
        opcodes_repo_state = repo_state(opcodes_root)
        spike_repo_state = repo_state(spike_root)
        spike_path_state = path_state(spike_root)
        checks = [
            (
                target_root,
                "riscv-gnu-toolchain repo",
                repo_link("riscv/riscv-gnu-toolchain"),
                toolchain_repo_state,
            ),
            (
                opcodes_root,
                "riscv-opcodes repo",
                repo_link("riscv/riscv-opcodes"),
                opcodes_repo_state,
            ),
            (rv_xcustom_path, "riscv-opcodes/extensions/rv_xcustom"),
            (encoding_out_path, "riscv-opcodes/encoding.out.h"),
            (instr_dict_path, "riscv-opcodes/instr_dict.json"),
            (
                spike_root,
                "spike repo",
                repo_link("riscv-software-src/riscv-isa-sim"),
                spike_repo_state,
            ),
            (spike_encoding_h, "spike/riscv/encoding.h"),
        ]

        for item in checks:
            if len(item) == 4:
                path_obj, label, link, repo_state_value = item
                state = "present" if repo_state_value == "present" else "missing"
                if repo_state_value == "not-a-repo":
                    state = "warning"
                results.append((label, state, str(path_obj), link, repo_state_value))
            else:
                path_obj, label = item
                state = path_state(path_obj)
                results.append((label, state, str(path_obj), None, None))

        print("riscv-opcodes preflight:")
        for label, state, path_str, link, repo_state_value in results:
            icon = status_icon(state)
            if link:
                print(f"  - {icon} {label}: {state} ({path_str})")
                print(f"    repo: {link}")
            else:
                print(f"  - {icon} {label}: {state} ({path_str})")

        problems = [item for item in results if item[1] != "present"]
        if not problems:
            print("OK: riscv-opcodes checkout looks usable")
            return 0
        unresolved = problems

        fix_actions = []
        if toolchain_repo_state == "present":
            if opcodes_repo_state == "missing":
                fix_actions.append((
                    opcodes_root.parent,
                    ["git", "clone", "https://github.com/riscv/riscv-opcodes", str(opcodes_root)],
                    "clone a standalone riscv-opcodes checkout next to the toolchain",
                ))
        elif toolchain_repo_state == "missing":
            fix_actions.append((
                target_root.parent,
                ["git", "clone", "https://github.com/riscv/riscv-gnu-toolchain", str(target_root)],
                "clone the missing riscv-gnu-toolchain repo",
            ))
        else:
            print(f"  - ⚠️ riscv-gnu-toolchain directory exists but is not a git repository: {target_root}")

        if toolchain_repo_state == "present":
            if spike_path_state in ("missing", "empty"):
                if spike_repo_url:
                    fix_actions.append((
                        target_root.parent,
                        ["git", "clone", spike_repo_url, str(spike_root)],
                        "clone the missing spike repo from .gitmodules",
                    ))
                else:
                    print("  - ⚠️ spike is missing and no URL was found in .gitmodules")
            elif spike_repo_state == "not-a-repo":
                print(f"  - ⚠️ spike directory exists but is not a git repository: {spike_root}")
                print(f"    repo: {repo_link('riscv-software-src/riscv-isa-sim')}")

        if opcodes_repo_state == "present" and not rv_xcustom_path.exists():
            if not rv_xcustom_parent.exists():
                fix_actions.append((
                    opcodes_root,
                    ["mkdir", "-p", str(rv_xcustom_parent)],
                    "create the extensions directory for rv_xcustom",
                ))
            fix_actions.append((
                opcodes_root,
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(rv_xcustom_path)!r}).write_text({RV_XCUSTOM_HEADER!r}, encoding='utf-8')"
                    ),
                ],
                "create rv_xcustom with the generated header",
            ))

        if opcodes_repo_state == "present" and (not encoding_out_path.exists() or not instr_dict_path.exists()):
            fix_actions.append((
                opcodes_root,
                ["make", "encoding.out.h"],
                "generate encoding.out.h and instr_dict.json from riscv-opcodes",
            ))
        elif opcodes_repo_state == "not-a-repo":
            print(f"  - ⚠️ riscv-opcodes directory exists but is not a git repository: {opcodes_root}")
            print(f"    repo: {repo_link('riscv/riscv-opcodes')}")

        if spike_repo_state == "present" and not spike_encoding_h.exists():
            print(f"  - ❌ missing spike encoding header: {spike_encoding_h}")
            print(f"    repo: {repo_link('riscv-software-src/riscv-isa-sim')}")
            fix_actions.append((
                target_root,
                ["git", "submodule", "sync", "--recursive", "--", "spike"],
                "sync the spike submodule metadata from .gitmodules",
            ))
            fix_actions.append((
                target_root,
                ["git", "submodule", "update", "--init", "--recursive", "--no-recommend-shallow", "spike"],
                "reinitialize the spike submodule so its tracked headers are checked out",
            ))

        print("\nSuggested fix:")
        if fix_actions:
            for cwd, cmd, desc in fix_actions:
                print(f"  - {desc}")
                print(f"    {' '.join(cmd)}")
        else:
            print("  - no automatic fix actions available")

        if fix and fix_actions:
            summary = "; ".join(desc for _, _, desc in fix_actions)
            print(f"\nFix summary: {summary}")

        if fix and not assume_yes:
            response = input("Apply preflight fixes for riscv-opcodes now? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                print("Aborted.")
                return 1

        if not fix or not fix_actions:
            return 1

        print("\nApplying fixes:")
        for cwd, cmd, desc in fix_actions:
            print(f"  - {desc}")
            print(f"    {' '.join(cmd)}")
            print("    running now...")
            subprocess.run(cmd, cwd=str(cwd), check=True)

        print("\nRe-running preflight after fixes...")

    print(f"Preflight still failing after {max_attempts} attempts.")
    print("Remaining issues:")
    for label, state, path_str, _, _ in unresolved:
        print(f"  - {label}: {state} ({path_str})")
    return 1


def find_used_slots(registry, opcode):
    used = []
    if not registry:
        return used
    for e in registry.get("entries", []):
        if e.get("opcode") == opcode and e.get("status") in ("allocated", "applied", "tombstoned"):
            used.append(e)
    return used


def allocate_for_format(fmt, used_entries):
    # returns dict of assigned fields (funct3, funct7 as applicable)
    if fmt == "R":
        used_pairs = set((e.get("encoding", {}).get("funct3"), e.get("encoding", {}).get("funct7")) for e in used_entries)
        for f3 in range(0, 8):
            for f7 in range(0, 128):
                if (f3, f7) not in used_pairs:
                    return {"funct3": f3, "funct7": f7}
        raise SystemExit("Exhausted R-type encodings for this opcode")
    if fmt in ("I", "S", "B"):
        used_f3 = set(e.get("encoding", {}).get("funct3") for e in used_entries)
        for f3 in range(0, 8):
            if f3 not in used_f3:
                return {"funct3": f3}
        raise SystemExit(f"Exhausted funct3 values for format {fmt}")
    if fmt in ("U", "J"):
        # opcode-only formats: require opcode-level uniqueness
        if used_entries:
            raise SystemExit(f"Opcode already used; cannot place {fmt}-type into this opcode")
        return {}
    raise SystemExit(f"Unsupported format: {fmt}")


def behavior_file_template(fmt_name):
    templates = {
        "R": "WRITE_RD(sext_xlen(RS1 + RS2));\n",
    }
    return templates.get(fmt_name, f"// TODO: implement {fmt_name}-type behavior\n")


def _ensure_behavior_file(base, behavior_relpath, name, fmt_name, opcode, encoding):
    """Create a minimal header file for the instruction on first allocation.

    Only creates the file if it does not already exist. The file will contain
    a basic format-specific body so generated Spike headers are immediately
    usable and easy to extend later.
    """
    try:
        bf_path = os.path.join(base, behavior_relpath)
        bf_dir = os.path.dirname(bf_path)
        if bf_dir and not os.path.exists(bf_dir):
            os.makedirs(bf_dir, exist_ok=True)

        # Only create on first allocation (i.e., when file does not exist)
        if os.path.exists(bf_path):
            return False

        with open(bf_path, "w", encoding="utf-8") as f:
            f.write(behavior_file_template(fmt_name))

        return True
    except Exception as e:
        print(f"   ⚠️  Failed to create behavior file '{behavior_relpath}': {e}")
        return False


def _behavior_file_path(base, behavior_relpath):
    return os.path.join(base, behavior_relpath)


def _default_behavior_relpath(name):
    return os.path.join("behavior_files", f"{name}.h")


def cmd_purge(args):
    base = os.path.dirname(__file__)
    config_dir = os.path.join(base, "config")
    registry_path = args.registry or os.path.join(config_dir, "instruction_registry.json")

    registry = load_json(registry_path)
    if not registry:
        print("No registry found")
        return

    purge_statuses = {"tombstoned", "removed"}
    entries = list(registry.get("entries", []))
    purge_entries = [entry for entry in entries if entry.get("status") in purge_statuses]

    if not purge_entries:
        print("No purgeable entries found")
        return

    remaining_entries = [entry for entry in entries if entry.get("status") not in purge_statuses]
    remaining_behavior_paths = {
        _behavior_file_path(base, entry.get("behavior_file", _default_behavior_relpath(str(entry.get("name", "")))))
        for entry in remaining_entries
    }

    plan = {
        "registry_path": registry_path,
        "purge": [],
    }
    for entry in purge_entries:
        behavior_relpath = entry.get("behavior_file", _default_behavior_relpath(entry.get("name", "unknown")))
        behavior_path = _behavior_file_path(base, behavior_relpath)
        plan["purge"].append(
            {
                "name": entry.get("name"),
                "status": entry.get("status"),
                "behavior_file": behavior_relpath,
                "behavior_file_path": behavior_path,
                "remove_behavior_file": behavior_path not in remaining_behavior_paths,
            }
        )

    print(json.dumps(plan, indent=2))

    if not args.apply:
        return

    purge_names = {entry["name"] for entry in plan["purge"] if entry.get("name")}
    registry["entries"] = [entry for entry in entries if entry.get("name") not in purge_names]

    for item in plan["purge"]:
        if not item["remove_behavior_file"]:
            continue
        behavior_path = item["behavior_file_path"]
        if os.path.exists(behavior_path):
            os.remove(behavior_path)
            print(f"Removed behavior file: {item['behavior_file']}")

    write_atomic(registry_path, registry)
    print(f"Purged {len(plan['purge'])} entr{'y' if len(plan['purge']) == 1 else 'ies'} from {registry_path}")


def cmd_allocate(args):
    base = os.path.dirname(__file__)
    config_dir = os.path.join(base, "config")
    behavior_dir = os.path.join(base, "behavior_files")
    fmt_path = args.format_registry or os.path.join(config_dir, "format_registry.json")
    manifest_path = args.manifest or os.path.join(base, "instructions.yaml")
    registry_path = args.registry or os.path.join(config_dir, "instruction_registry.json")

    fmt = load_json(fmt_path)
    if fmt is None:
        raise SystemExit(f"Cannot find format registry at {fmt_path}")
    manifest = parse_manifest(manifest_path)
    registry = load_json(registry_path) or {"version": 1, "entries": []}
    registry_by_name = registry_entry_map(registry)

    errors, warnings = validate_manifest(manifest, fmt)
    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit("Manifest validation failed")
    working_registry = copy.deepcopy(registry)
    instructions = manifest.get("instructions") or []

    manifest_name_set = set(manifest_names(manifest))
    if not manifest_name_set and registry_by_name:
        warnings.append("manifest contains no instructions; all registry entries will be graveyarded on apply")
    for name, entry in registry_by_name.items():
        if name in manifest_name_set:
            warnings.append(
                f"{name}: already exists in registry (status={entry.get('status', 'unknown')}); will be kept, not reallocated"
            )
        else:
            warnings.append(
                f"{name}: not present in instructions.yaml; will be graveyarded (status={entry.get('status', 'unknown')})"
            )

    plan = {"keep": [], "allocate": [], "graveyard": []}

    # sort instructions deterministic by name
    for instr in sorted(instructions, key=lambda x: x.get("name", "")):
        name = instr["name"]
        fmt_name = instr["format"]
        if name in registry_by_name:
            existing_entry = registry_by_name[name]
            if existing_entry.get("status") == "tombstoned":
                warnings.append(
                    f"{name}: tombstoned in registry but present in instructions.yaml; will be untombstoned"
                )
            else:
                warnings.append(
                    f"{name}: already exists in registry (status={existing_entry.get('status', 'unknown')}); will be kept, not reallocated"
                )
            status = "allocated" if existing_entry.get("status") == "tombstoned" else existing_entry.get("status", "allocated")
            plan["keep"].append(
                ordered_registry_entry(
                    name,
                    fmt_name,
                    existing_entry.get("opcode"),
                    existing_entry.get("behavior_file", os.path.join("behavior_files", f"{name}.h")),
                    existing_entry.get("encoding", {}),
                    status,
                    existing_entry.get("source", f"instructions.yaml#{name}"),
                    existing_entry.get("allocated_at", datetime.utcnow().isoformat() + "Z"),
                )
            )
            continue
        default_opcode = fmt.get("default_opcode_per_format", {}).get(fmt_name)
        if default_opcode is None:
            raise SystemExit(f"No default opcode for format {fmt_name} in format registry")
        used = find_used_slots(working_registry, default_opcode)
        assigned = allocate_for_format(fmt_name, used)
        behavior_file = os.path.join("behavior_files", f"{name}.h")
        proposal = {
            "name": name,
            "format": fmt_name,
            "opcode": default_opcode,
            "behavior_file": behavior_file,
            "encoding": assigned,
            "source": f"instructions.yaml#{name}",
        }
        plan["allocate"].append(proposal)

        # Reserve the slot in-memory so later proposals in the same run cannot
        # reuse the same opcode+funct combination.
        working_registry.setdefault("entries", []).append(
            ordered_registry_entry(
                name,
                fmt_name,
                default_opcode,
                behavior_file,
                assigned,
                "allocated",
                proposal["source"],
                datetime.utcnow().isoformat() + "Z",
            )
        )

    for name, entry in registry_by_name.items():
        if name not in manifest_name_set:
            plan["graveyard"].append(
                ordered_registry_entry(
                    name,
                    entry.get("format"),
                    entry.get("opcode"),
                    entry.get("behavior_file", os.path.join("behavior_files", f"{name}.h")),
                    entry.get("encoding", {}),
                    "tombstoned",
                    entry.get("source", f"instructions.yaml#{name}"),
                    entry.get("allocated_at", datetime.utcnow().isoformat() + "Z"),
                )
            )

    for warning in warnings:
        print("INFO:", warning)

    # print proposals
    print(json.dumps(plan, indent=2))

    if args.apply:
        keep_names = {p["name"] for p in plan["keep"]}
        for p in plan["allocate"]:
            entry = ordered_registry_entry(
                p["name"],
                p["format"],
                p["opcode"],
                p["behavior_file"],
                p["encoding"],
                "allocated",
                p.get("source"),
                datetime.utcnow().isoformat() + "Z",
            )
            registry.setdefault("entries", []).append(entry)

        for entry in registry.setdefault("entries", []):
            if entry.get("name") in keep_names and entry.get("status") == "tombstoned":
                entry["status"] = "allocated"

        for p in plan["graveyard"]:
            for entry in registry.setdefault("entries", []):
                if entry.get("name") == p["name"]:
                    if entry.get("status") != "tombstoned":
                        entry["status"] = "tombstoned"
                    break
        write_atomic(registry_path, registry)
        print(
            f"Wrote {registry_path} ({len(plan['allocate'])} allocated, {len(plan['graveyard'])} graveyarded, {len(plan['keep'])} kept)"
        )
        # Generate behavior files for newly allocated instructions (only on apply)
        for p in plan.get("allocate", []):
            name = p.get("name")
            fmt_name = p.get("format")
            opcode = p.get("opcode")
            encoding = p.get("encoding", {})
            behavior_rel = p.get("behavior_file", os.path.join("behavior_files", f"{name}.h"))

            created = _ensure_behavior_file(base, behavior_rel, name, fmt_name, opcode, encoding)
            if created:
                print(f"   ✅ Created behavior file '{behavior_rel}'")


def cmd_validate(args):
    base = os.path.dirname(__file__)
    config_dir = os.path.join(base, "config")
    fmt_path = args.format_registry or os.path.join(config_dir, "format_registry.json")
    manifest_path = args.manifest or os.path.join(base, "instructions.yaml")
    fmt = load_json(fmt_path)
    if fmt is None:
        raise SystemExit(f"Cannot find format registry at {fmt_path}")
    registry = load_json(os.path.join(config_dir, "instruction_registry.json"))
    manifest = parse_manifest(manifest_path)
    errors, warnings = validate_manifest(manifest, fmt)
    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(2)
    registry_by_name = registry_entry_map(registry)
    manifest_name_set = set(manifest_names(manifest))
    for name, entry in registry_by_name.items():
        if name in manifest_name_set:
            if entry.get("status") == "tombstoned":
                print(f"INFO: {name} is tombstoned in registry but present in instructions.yaml; will be untombstoned")
            else:
                print(
                    f"INFO: {name} already exists in registry (status={entry.get('status', 'unknown')}); will not be reallocated"
                )
        else:
            print(
                f"INFO: {name} exists in registry but not in instructions.yaml; would be graveyarded on apply"
            )
    for warning in warnings:
        print("INFO:", warning)
    print("Manifest OK")


def cmd_show(args):
    base = os.path.dirname(__file__)
    config_dir = os.path.join(base, "config")
    registry_path = args.registry or os.path.join(config_dir, "instruction_registry.json")
    registry = load_json(registry_path)
    if not registry:
        print("No registry found")
        return
    if args.name:
        for e in registry.get("entries", []):
            if e.get("name") == args.name:
                print(json.dumps(e, indent=2))
                return
        print(f"No entry named {args.name}")
        return
    print(json.dumps(registry, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="riscv-ci", description="RISC-V custom instruction allocator CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_alloc = sub.add_parser("allocate")
    p_alloc.add_argument("--dry-run", action="store_true")
    p_alloc.add_argument("--apply", action="store_true")
    p_alloc.add_argument("--format-registry", dest="format_registry")
    p_alloc.add_argument("--manifest", dest="manifest")
    p_alloc.add_argument("--registry", dest="registry")
    p_alloc.set_defaults(func=cmd_allocate)

    p_val = sub.add_parser("validate")
    p_val.add_argument("--format-registry", dest="format_registry")
    p_val.add_argument("--manifest", dest="manifest")
    p_val.set_defaults(func=cmd_validate)

    p_show = sub.add_parser("show")
    p_show.add_argument("--registry", dest="registry")
    p_show.add_argument("--name", dest="name")
    p_show.set_defaults(func=cmd_show)

    p_purge = sub.add_parser("purge")
    p_purge.add_argument("--apply", action="store_true", help="apply purge changes")
    p_purge.add_argument("--registry", dest="registry")
    p_purge.set_defaults(func=cmd_purge)

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--fix", action="store_true", help="attempt to repair missing riscv-opcodes and spike pieces")
    p_doctor.add_argument("--yes", action="store_true", help="apply fixes without prompting")
    p_doctor.add_argument("--toolchain-root", dest="toolchain_root", help="path to the riscv-gnu-toolchain checkout")
    p_doctor.set_defaults(
        func=lambda args: sys.exit(
            doctor_riscv_opcodes(
                os.path.dirname(__file__),
                fix=args.fix,
                assume_yes=args.yes,
                toolchain_root_override=args.toolchain_root,
            )
        )
    )

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
