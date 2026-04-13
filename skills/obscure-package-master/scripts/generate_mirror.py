import ast
import os
import json
import sys
import subprocess
import tarfile
import zipfile
import shutil
import glob
import site

# Known default skill-installation paths per major AI agent provider.
# These can be overridden via config.json or the AGENT_SKILLS_PATH env var.
PROVIDER_DEFAULTS = {
    "claude":   "~/.claude/skills",
    "gemini":   "~/.gemini/skills",
    "codex":    "~/.copilot/skills",
    "cursor":   "~/.cursor/skills",
    "openai":   "~/.openai/skills",
    "openclaw": "~/.openclaw/skills",
    "cline":    "~/.cline/skills",
}

def get_docstring_summary(node):
    docstring = ast.get_docstring(node)
    if docstring:
        return docstring.split('\n')[0].strip()
    return ""

def parse_file(file_path, package_root):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            content = f.read()
            tree = ast.parse(content)
        except Exception:
            return []
    
    relative_path = os.path.relpath(file_path, package_root)
    results = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                kind = "class"
                bases = [ast.unparse(b) for b in node.bases]
                signature = f"class {node.name}({', '.join(bases)})"
            else:
                kind = "function"
                args = ast.unparse(node.args)
                signature = f"def {node.name}({args})"

            start_line = node.lineno
            end_line = node.end_lineno
            doc_summary = get_docstring_summary(node)
            
            results.append({
                "name": node.name,
                "kind": kind,
                "signature": signature,
                "doc": doc_summary,
                "file": relative_path,
                "range": [start_line, end_line]
            })
            
    return results

def _safe_extract_tar(tar, path):
    """Extract a tarfile archive while rejecting members that would escape *path*."""
    real_path = os.path.realpath(path) + os.sep
    safe_members = []
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(path, member.name))
        if not member_path.startswith(real_path):
            print(f"Warning: Skipping unsafe archive member: {member.name!r}")
            continue
        safe_members.append(member)
    # Pass filter='data' when available (Python 3.12+) to suppress the
    # DeprecationWarning about unfiltered extraction in future Python versions.
    try:
        tar.extractall(path=path, members=safe_members, filter="data")
    except TypeError:
        tar.extractall(path=path, members=safe_members)


def _safe_extract_zip(zf, path):
    """Extract a zipfile archive while rejecting members that would escape *path*."""
    real_path = os.path.realpath(path) + os.sep
    for name in zf.namelist():
        member_path = os.path.realpath(os.path.join(path, name))
        if not member_path.startswith(real_path):
            print(f"Warning: Skipping unsafe archive member: {name!r}")
            continue
        zf.extract(name, path)


def download_package(package, version, tmp_dir):
    print(f"Downloading {package}=={version}...")
    cmd = ["pip", "download", "--no-binary=:all:", f"{package}=={version}", "-d", tmp_dir]
    subprocess.run(cmd, check=True)
    
    # Find the downloaded file that best matches the package name
    files = os.listdir(tmp_dir)
    # Filter for files that contain the package name and are archives
    matching_files = [f for f in files if package.replace("-", "_") in f.lower().replace("-", "_") and (f.endswith(".tar.gz") or f.endswith(".zip") or f.endswith(".whl"))]
    
    if not matching_files:
        # Fallback to the first one if we can't find a perfect match
        archive_path = os.path.join(tmp_dir, files[0])
    else:
        # Pick the one that most likely corresponds to the package (e.g. shortest name if multiple)
        matching_files.sort(key=len)
        archive_path = os.path.join(tmp_dir, matching_files[0])
    
    print(f"Extracting {archive_path}...")
    extract_path = os.path.join(tmp_dir, "extracted")
    os.makedirs(extract_path, exist_ok=True)
    
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tar.bz2"):
        with tarfile.open(archive_path) as tar:
            _safe_extract_tar(tar, extract_path)
    elif archive_path.endswith(".zip") or archive_path.endswith(".whl"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            _safe_extract_zip(zip_ref, extract_path)
            
    return extract_path

def find_package_root(extract_path, package_name):
    # Normalize package name (PyPI uses hyphens and underscores interchangeably sometimes)
    norm_name = package_name.lower().replace("-", "_")
    
    # Search patterns
    # We look for a directory that contains an __init__.py and matches our normalized name
    for root, dirs, files in os.walk(extract_path):
        for d in dirs:
            if d.lower().replace("-", "_") == norm_name:
                pkg_dir = os.path.join(root, d)
                if os.path.exists(os.path.join(pkg_dir, "__init__.py")):
                    return pkg_dir
    
    return None

def _get_site_packages_candidates():
    """Return site-packages directories to search for installed packages.

    Includes the current Python environment's site-packages and any common
    virtualenv directories (.venv, venv, env, .env) found directly inside the
    current working directory.  No recursive filesystem walk is performed so
    this stays fast regardless of project size.
    """
    candidates = []

    # 1. Current Python interpreter's site-packages
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        # getsitepackages() is absent in some virtualenv builds
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site not in candidates:
            candidates.append(user_site)
    except AttributeError:
        pass

    # 2. Common virtualenv directory names in the current working directory
    for venv_name in (".venv", "venv", "env", ".env"):
        venv_root = os.path.join(os.getcwd(), venv_name)

        # Unix layout: {venv}/lib/pythonX.Y/site-packages
        lib_path = os.path.join(venv_root, "lib")
        if os.path.isdir(lib_path):
            try:
                for entry in os.listdir(lib_path):
                    sp = os.path.join(lib_path, entry, "site-packages")
                    if os.path.isdir(sp):
                        candidates.append(sp)
            except OSError:
                pass

        # Windows layout: {venv}/Lib/site-packages (capital L, no version subfolder)
        win_sp = os.path.join(venv_root, "Lib", "site-packages")
        if os.path.isdir(win_sp) and win_sp not in candidates:
            candidates.append(win_sp)

    return candidates


def find_local_package(package_name, version):
    """Check whether *package_name==version* is already installed locally.

    Searches the current Python environment and any common virtualenv
    directories (.venv, venv, env, .env) in the current working directory.
    No full-filesystem scan is performed; once a dist-info entry that
    matches the requested version is found, find_package_root() walks only
    that site-packages directory to locate the package source.

    Returns the path to the package source directory (the directory that
    contains ``__init__.py``) if a matching installation is found, or
    ``None`` otherwise.
    """
    norm_name = package_name.lower().replace("-", "_")
    hyphen_name = package_name.lower().replace("_", "-")
    # Both common normalizations for the dist-info directory name
    dist_info_names = {
        f"{norm_name}-{version}.dist-info",
        f"{hyphen_name}-{version}.dist-info",
    }

    for site_pkg in _get_site_packages_candidates():
        if not os.path.isdir(site_pkg):
            continue
        try:
            entries = os.listdir(site_pkg)
        except OSError:
            continue
        # A dist-info directory with the exact version confirms the installation
        has_dist_info = any(e.lower() in dist_info_names for e in entries)
        if has_dist_info:
            pkg_dir = find_package_root(site_pkg, package_name)
            if pkg_dir:
                return pkg_dir

    return None


def build_grep_map(package_root):
    grep_map = []
    # We want to maintain relative structure from the root
    for root, _, files in os.walk(package_root):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Skip __pycache__ etc
                if "__pycache__" in file_path:
                    continue
                grep_map.extend(parse_file(file_path, package_root))
    return grep_map

def _resolve_skills_dir(explicit_path, local, config_skills_path):
    """Return the final output directory for the generated skill, or ``None``.

    Priority (highest first):
    1. *explicit_path* — a path supplied directly on the CLI.
    2. *local* flag — installs directly into the current working directory.
    3. *config_skills_path* — whatever ``get_config()`` resolved (env var,
       config file, or provider default).  May be ``None`` if nothing was
       configured.

    Returns ``None`` when no directory could be determined; the caller is
    responsible for reporting an error in that case.
    """
    if explicit_path is not None:
        return os.path.expanduser(explicit_path)
    if local:
        return os.getcwd()
    if config_skills_path is not None:
        return os.path.expanduser(config_skills_path)
    return None


def generate_skill(package, version, grep_map, package_root, output_dir, use_symlinks=False):
    skill_name = f"{package}-{version}".lower().replace("_", "-")
    skill_dir = os.path.join(output_dir, skill_name)
    os.makedirs(skill_dir, exist_ok=True)
    
    ref_dir = os.path.join(skill_dir, "references")
    os.makedirs(ref_dir, exist_ok=True)
    
    # Copy (or symlink) source files to references/
    for item in grep_map:
        src_file = os.path.join(package_root, item['file'])
        dst_file = os.path.join(ref_dir, item['file'])
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        if not os.path.exists(dst_file):
            try:
                if use_symlinks:
                    try:
                        rel_src = os.path.relpath(
                            os.path.abspath(src_file), os.path.dirname(dst_file)
                        )
                        os.symlink(rel_src, dst_file)
                    except (ValueError, OSError, NotImplementedError):
                        # Fall back to a regular copy when symlinks are unavailable
                        # (e.g. Windows without Developer Mode, or cross-drive paths).
                        shutil.copy2(src_file, dst_file)
                else:
                    shutil.copy2(src_file, dst_file)
            except Exception as e:
                op = "symlinking (with copy fallback)" if use_symlinks else "copying"
                print(f"Error {op} {src_file}: {e}")

    # Build SKILL.md
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_md_path, "w") as f:
        f.write(f"---\nname: {skill_name}\ndescription: Deterministic API mirror for {package} v{version}. Use this skill to grounded exploration of {package} API when uncertainty is high.\n---\n\n")
        f.write(f"# {package} v{version} Grep Map\n\n")
        f.write("This map provides a deterministic coordinate system for the package source. All paths are relative to `references/`.\n\n")
        
        # Organize by file
        by_file = {}
        for item in grep_map:
            by_file.setdefault(item['file'], []).append(item)
            
        for file_path, items in sorted(by_file.items()):
            f.write(f"## File: `{file_path}`\n\n")
            # Sort items by line range to be deterministic
            items.sort(key=lambda x: x['range'][0])
            for item in items:
                f.write(f"- **`{item['signature']}`**\n")
                f.write(f"  - Range: `lines {item['range'][0]}-{item['range'][1]}`\n")
                if item['doc']:
                    f.write(f"  - Doc: {item['doc']}\n")
            f.write("\n")

    print(f"Skill generated at: {skill_dir}")
    return skill_dir

def detect_provider(provider_defaults=None, script_path=None):
    """Detect the active AI agent provider without reading sensitive env vars.

    Detection order:
    1. ``AGENT_PROVIDER`` environment variable (explicit, purpose-built).
    2. The directory the script is installed in — if it lives inside a known
       provider's skills folder (e.g. ``~/.claude/skills/…``) the provider is
       inferred from the path.  This works regardless of the authentication
       method (API key, OAuth, etc.) and avoids touching any credentials.
    3. Returns ``None`` if neither heuristic matches.
    """
    # Explicit override always wins; AGENT_PROVIDER is purpose-built and safe.
    if "AGENT_PROVIDER" in os.environ:
        return os.environ["AGENT_PROVIDER"].lower()

    # Infer provider from the script's installed location.
    # Expected layout: <skills_dir>/<skill_name>/scripts/generate_mirror.py
    # → skills_dir is two levels above the scripts/ directory.
    if provider_defaults is None:
        provider_defaults = PROVIDER_DEFAULTS
    if script_path is None:
        script_path = os.path.abspath(__file__)

    # Use normpath (not realpath) to canonicalise — realpath calls abspath
    # internally which can interfere with test mocking.
    skills_dir = os.path.normpath(
        os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    )
    for provider, default_path in provider_defaults.items():
        if skills_dir == os.path.normpath(os.path.expanduser(default_path)):
            return provider

    return None


def get_config():
    # Compute script path once so detect_provider uses the same value.
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    config_path = os.path.join(os.path.dirname(script_dir), "config.json")

    config = {
        "provider": None,
        "provider_defaults": PROVIDER_DEFAULTS.copy(),
    }

    user_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            # Deep-merge provider_defaults so callers can extend the table
            if "provider_defaults" in user_config:
                config["provider_defaults"].update(user_config.pop("provider_defaults"))
            config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not read config.json: {e}")

    # Resolve provider: AGENT_PROVIDER env var and install-path detection,
    # then fall back to the provider set in config.json.
    provider = detect_provider(
        provider_defaults=config["provider_defaults"],
        script_path=script_path,
    ) or config.get("provider")
    if provider:
        config["provider"] = provider

    # Apply provider default path only when no explicit path was configured
    explicit_path_in_config = "skills_path" in user_config
    if provider and provider in config["provider_defaults"] and not explicit_path_in_config:
        config["skills_path"] = os.path.expanduser(config["provider_defaults"][provider])

    # If no skills_path was determined, leave it absent — callers must handle this.

    # Environment variable override – highest priority
    if "AGENT_SKILLS_PATH" in os.environ:
        config["skills_path"] = os.environ["AGENT_SKILLS_PATH"]

    return config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="generate_mirror.py",
        description="Generate a local grep-map skill for a PyPI package.",
    )
    parser.add_argument("package", help="PyPI package name")
    parser.add_argument("version", help="Package version (e.g. 1.2.3)")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help=(
            "Explicit output directory for the skill. "
            "Overrides --local, config, and provider auto-detection."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Install the skill into the current working directory, "
            "bypassing provider auto-detection and config. "
            "Useful for project-scoped skills."
        ),
    )
    args = parser.parse_args()

    pkg = args.package
    ver = args.version

    config = get_config()
    skills_dir = _resolve_skills_dir(args.output_path, args.local, config.get("skills_path"))

    if skills_dir is None:
        print(
            "Error: Could not determine an output directory.\n"
            "Options:\n"
            "  --local                   Install into the current working directory\n"
            "  AGENT_SKILLS_PATH=<path>  Set via environment variable\n"
            "  skills_path in config.json\n"
            "  python3 generate_mirror.py <pkg> <ver> <output_path>"
        )
        sys.exit(1)

    # Check for a local installation before downloading
    print(f"Checking for local installation of {pkg}=={ver}...")
    local_pkg_root = find_local_package(pkg, ver)

    if local_pkg_root:
        print(f"Found local installation at: {local_pkg_root}")
        try:
            g_map = build_grep_map(local_pkg_root)
            generate_skill(pkg, ver, g_map, local_pkg_root, skills_dir, use_symlinks=True)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        tmp = "tmp_download"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)

        try:
            extract_path = download_package(pkg, ver, tmp)
            pkg_root = find_package_root(extract_path, pkg)

            if not pkg_root:
                print(f"Could not find package root for {pkg} in {extract_path}")
                sys.exit(1)

            print(f"Found package root at: {pkg_root}")
            g_map = build_grep_map(pkg_root)
            generate_skill(pkg, ver, g_map, pkg_root, skills_dir)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
