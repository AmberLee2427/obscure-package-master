"""
Tests for skills/obscure-package-master/scripts/generate_mirror.py

All external I/O (subprocess / pip download) is mocked so that the suite
runs offline and on every major AI-agent provider environment (Claude,
Gemini, Codex, Cursor, OpenAI, OpenClaw, Cline, …).

Matrix of OS × Python versions is driven by the CI workflow
(.github/workflows/tests.yml).
"""

import ast
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def _check_symlinks_available():
    """Return True if the OS permits creating symlinks without special privileges."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.txt")
        dst = os.path.join(td, "dst.txt")
        with open(src, "w"):
            pass
        try:
            os.symlink(src, dst)
            return True
        except (OSError, NotImplementedError):
            return False


SYMLINKS_AVAILABLE = _check_symlinks_available()
requires_symlinks = pytest.mark.skipif(
    not SYMLINKS_AVAILABLE,
    reason="Symlinks require Developer Mode or elevated privileges on this platform",
)

# ---------------------------------------------------------------------------
# Helpers – load the module under test without executing __main__
# ---------------------------------------------------------------------------

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "obscure-package-master"
    / "scripts"
    / "generate_mirror.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_mirror", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gm = _load_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_pkg(tmp_path):
    """Create a minimal fake Python package on-disk for parsing tests."""
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        textwrap.dedent("""\
            \"\"\"Top-level package.\"\"\"

            class MyClass(object):
                \"\"\"A simple class.\"\"\"
                def method(self, x: int) -> int:
                    \"\"\"Double x.\"\"\"
                    return x * 2

            async def async_helper(name: str) -> None:
                \"\"\"Async helper.\"\"\"
                pass

            def _private():
                pass
        """)
    )
    sub = pkg_dir / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "utils.py").write_text(
        textwrap.dedent("""\
            def util_func(a, b):
                \"\"\"Add two numbers.\"\"\"
                return a + b

            class Helper:
                pass
        """)
    )
    return pkg_dir


@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all provider-related env vars before each test."""
    for var in (
        "AGENT_PROVIDER",
        "AGENT_SKILLS_PATH",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVEAI_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "GITHUB_COPILOT_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ===========================================================================
# 1. get_docstring_summary
# ===========================================================================

class TestGetDocstringSummary:
    def test_returns_first_line(self):
        src = textwrap.dedent("""\
            def f():
                \"\"\"First line.
                Second line.
                \"\"\"
                pass
        """)
        node = ast.parse(src).body[0]
        assert gm.get_docstring_summary(node) == "First line."

    def test_returns_empty_string_when_no_docstring(self):
        src = "def f(): pass"
        node = ast.parse(src).body[0]
        assert gm.get_docstring_summary(node) == ""

    def test_strips_whitespace(self):
        src = 'def f():\n    """   Leading spaces.   """\n    pass'
        node = ast.parse(src).body[0]
        assert gm.get_docstring_summary(node) == "Leading spaces."


# ===========================================================================
# 2. parse_file
# ===========================================================================

class TestParseFile:
    def test_finds_class_and_methods(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        names = {r["name"] for r in results}
        assert "MyClass" in names
        assert "method" in names

    def test_finds_async_function(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        names = {r["name"] for r in results}
        assert "async_helper" in names

    def test_class_kind(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        cls = next(r for r in results if r["name"] == "MyClass")
        assert cls["kind"] == "class"

    def test_function_kind(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        fn = next(r for r in results if r["name"] == "_private")
        assert fn["kind"] == "function"

    def test_line_range_is_plausible(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        cls = next(r for r in results if r["name"] == "MyClass")
        start, end = cls["range"]
        assert start >= 1
        assert end >= start

    def test_relative_file_path(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        assert all(not r["file"].startswith(str(tmp_pkg)) for r in results)

    def test_doc_captured(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        cls = next(r for r in results if r["name"] == "MyClass")
        assert cls["doc"] == "A simple class."

    def test_returns_empty_list_for_syntax_error(self, tmp_path):
        bad_py = tmp_path / "bad.py"
        bad_py.write_text("def (:")
        results = gm.parse_file(str(bad_py), str(tmp_path))
        assert results == []

    def test_signature_format_function(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        fn = next(r for r in results if r["name"] == "async_helper")
        assert fn["signature"].startswith("def async_helper(")

    def test_signature_format_class(self, tmp_pkg):
        init_py = tmp_pkg / "__init__.py"
        results = gm.parse_file(str(init_py), str(tmp_pkg))
        cls = next(r for r in results if r["name"] == "MyClass")
        assert cls["signature"].startswith("class MyClass(")


# ===========================================================================
# 3. find_package_root
# ===========================================================================

class TestFindPackageRoot:
    def test_finds_root_by_name(self, tmp_pkg):
        extract = tmp_pkg.parent
        result = gm.find_package_root(str(extract), "mypkg")
        assert result is not None
        assert result.endswith("mypkg")

    def test_returns_none_when_not_found(self, tmp_path):
        result = gm.find_package_root(str(tmp_path), "nonexistent")
        assert result is None

    def test_case_insensitive_with_underscores(self, tmp_path):
        pkg = tmp_path / "my_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        result = gm.find_package_root(str(tmp_path), "My-Pkg")
        assert result is not None

    def test_ignores_dirs_without_init(self, tmp_path):
        # Dir exists but has no __init__.py → should not be found
        (tmp_path / "mypkg").mkdir()
        result = gm.find_package_root(str(tmp_path), "mypkg")
        assert result is None


# ===========================================================================
# 4. build_grep_map
# ===========================================================================

class TestBuildGrepMap:
    def test_returns_list(self, tmp_pkg):
        result = gm.build_grep_map(str(tmp_pkg))
        assert isinstance(result, list)

    def test_includes_sub_package_functions(self, tmp_pkg):
        result = gm.build_grep_map(str(tmp_pkg))
        names = {r["name"] for r in result}
        assert "util_func" in names
        assert "Helper" in names

    def test_skips_pycache(self, tmp_pkg):
        cache = tmp_pkg / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("def cached(): pass")
        result = gm.build_grep_map(str(tmp_pkg))
        names = {r["name"] for r in result}
        assert "cached" not in names


# ===========================================================================
# 5. generate_skill
# ===========================================================================

class TestGenerateSkill:
    def _run(self, tmp_path, tmp_pkg):
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = gm.generate_skill("mypkg", "1.0.0", grep_map, str(tmp_pkg), str(tmp_path))
        return Path(skill_dir)

    def test_creates_skill_dir(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        assert sd.exists()

    def test_creates_references_dir(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        assert (sd / "references").is_dir()

    def test_creates_skill_md(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        assert (sd / "SKILL.md").is_file()

    def test_skill_md_contains_package_name(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        content = (sd / "SKILL.md").read_text()
        assert "mypkg" in content

    def test_skill_md_contains_frontmatter(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        content = (sd / "SKILL.md").read_text()
        assert content.startswith("---")

    def test_skill_md_contains_line_ranges(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        content = (sd / "SKILL.md").read_text()
        assert "lines" in content

    def test_skill_dir_name_lowercased(self, tmp_path, tmp_pkg):
        grep_map = gm.build_grep_map(str(tmp_pkg))
        sd = Path(gm.generate_skill("MyPkg", "1.0.0", grep_map, str(tmp_pkg), str(tmp_path)))
        assert sd.name == "mypkg-1.0.0"

    def test_source_files_copied_to_references(self, tmp_path, tmp_pkg):
        sd = self._run(tmp_path, tmp_pkg)
        # __init__.py must be present in references/
        ref_init = sd / "references" / "__init__.py"
        assert ref_init.is_file()

    def test_empty_grep_map_creates_valid_skill(self, tmp_path, tmp_pkg):
        sd = Path(gm.generate_skill("mypkg", "0.0.1", [], str(tmp_pkg), str(tmp_path)))
        assert (sd / "SKILL.md").is_file()


# ===========================================================================
# 6. detect_provider
# ===========================================================================

class TestDetectProvider:
    def test_explicit_env_var(self, clean_env):
        clean_env.setenv("AGENT_PROVIDER", "Gemini")
        assert gm.detect_provider() == "gemini"

    def test_no_hints_returns_none(self, clean_env, tmp_path):
        # Script path inside a tmp dir that doesn't match any provider path
        assert gm.detect_provider(script_path=str(tmp_path / "scripts" / "generate_mirror.py")) is None

    def test_path_based_detection_claude(self, clean_env, tmp_path, monkeypatch):
        """Script installed under ~/.claude/skills/<skill>/scripts/ → 'claude'."""
        fake_skills = tmp_path / ".claude" / "skills"
        fake_skills.mkdir(parents=True)
        script = fake_skills / "obscure-package-master" / "scripts" / "generate_mirror.py"
        defaults = {"claude": str(fake_skills)}
        assert gm.detect_provider(provider_defaults=defaults, script_path=str(script)) == "claude"

    def test_path_based_detection_local_install(self, clean_env, tmp_path, monkeypatch):
        """Script in <cwd>/.claude/skills/<skill>/scripts/ is detected as 'claude'."""
        monkeypatch.chdir(tmp_path)
        # Local install places the skill under the CWD-relative provider hidden dir.
        local_skills = tmp_path / ".claude" / "skills"
        local_skills.mkdir(parents=True)
        script = local_skills / "obscure-package-master" / "scripts" / "generate_mirror.py"
        # Default points to the global path; detection must also match the local equivalent.
        global_skills_path = os.path.normpath(os.path.join(os.path.expanduser("~"), ".claude", "skills"))
        defaults = {"claude": global_skills_path}
        assert gm.detect_provider(provider_defaults=defaults, script_path=str(script)) == "claude"

    def test_path_based_unrecognised_location_returns_none(self, clean_env, tmp_path):
        script = tmp_path / "some" / "random" / "scripts" / "generate_mirror.py"
        assert gm.detect_provider(provider_defaults=gm.PROVIDER_DEFAULTS, script_path=str(script)) is None

    def test_explicit_env_var_overrides_path(self, clean_env, tmp_path):
        """AGENT_PROVIDER wins even when the path would suggest a different provider."""
        clean_env.setenv("AGENT_PROVIDER", "cursor")
        fake_skills = tmp_path / ".openai" / "skills"
        fake_skills.mkdir(parents=True)
        script = fake_skills / "obscure-package-master" / "scripts" / "generate_mirror.py"
        defaults = {"openai": str(fake_skills)}
        assert gm.detect_provider(provider_defaults=defaults, script_path=str(script)) == "cursor"

    def test_api_keys_are_never_read(self, clean_env):
        """Setting provider API key vars must not affect the result."""
        for var in (
            "ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_GENERATIVEAI_API_KEY", "OPENAI_API_KEY",
            "CODEX_API_KEY", "GITHUB_COPILOT_TOKEN",
        ):
            clean_env.setenv(var, "sk-dummy")
        # None of those should influence the outcome
        result = gm.detect_provider(script_path="/tmp/nowhere/scripts/generate_mirror.py")
        assert result is None


# ===========================================================================
# 7. get_config – various config set-ups
# ===========================================================================

class TestGetConfigDefaultOnly:
    """No config file, no env vars → defaults."""

    def test_skills_path_absent_when_no_hints(self, clean_env, tmp_path, monkeypatch):
        # Point script dir to a tmpdir so no config.json is found and no
        # provider path is matched.
        monkeypatch.setattr(gm, "__file__", str(tmp_path / "scripts" / "generate_mirror.py"))
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert "skills_path" not in cfg

    def test_provider_is_none_without_hints(self, clean_env, tmp_path):
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert cfg.get("provider") is None

    def test_provider_env_var_sets_default_path(self, clean_env):
        """Explicit AGENT_PROVIDER env var → use provider default path."""
        clean_env.setenv("AGENT_PROVIDER", "claude")
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert cfg["provider"] == "claude"
        assert cfg["skills_path"] == os.path.expanduser("~/.claude/skills")

    def test_api_keys_do_not_affect_config(self, clean_env):
        """API key env vars must never influence provider detection."""
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert cfg.get("provider") is None
        assert "skills_path" not in cfg


class TestGetConfigFromFile:
    """Config loaded from a config.json."""

    def _write_config(self, tmp_path, data):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(data))
        return cfg_file

    def test_skills_path_from_file(self, clean_env, tmp_path):
        self._write_config(tmp_path, {"skills_path": "/custom/skills"})
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        # Directly patch the script's __file__ resolution path
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["skills_path"] == "/custom/skills"

    def test_provider_from_file(self, clean_env, tmp_path):
        self._write_config(tmp_path, {"provider": "gemini", "skills_path": "/g/skills"})
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["provider"] == "gemini"

    def test_custom_provider_defaults_merged(self, clean_env, tmp_path):
        self._write_config(
            tmp_path,
            {
                "skills_path": "/s",
                "provider_defaults": {"mycorp": "~/.mycorp/skills"},
            },
        )
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert "mycorp" in cfg["provider_defaults"]
        # Built-in providers still present
        assert "claude" in cfg["provider_defaults"]

    def test_provider_default_path_applied_when_no_explicit_skills_path(self, clean_env, tmp_path):
        """When config sets a provider but no skills_path, the provider default path is used."""
        self._write_config(tmp_path, {"provider": "gemini"})
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["skills_path"] == os.path.expanduser("~/.gemini/skills")

    def test_provider_default_path_not_applied_when_skills_path_set(self, clean_env, tmp_path):
        """Explicit skills_path in config takes priority over provider default."""
        self._write_config(tmp_path, {"provider": "gemini", "skills_path": "/explicit/path"})
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["skills_path"] == "/explicit/path"

    def test_agent_provider_env_overrides_config_provider(self, clean_env, tmp_path):
        """AGENT_PROVIDER env var takes priority over provider set in config.json."""
        clean_env.setenv("AGENT_PROVIDER", "cursor")
        self._write_config(tmp_path, {"provider": "gemini"})
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["provider"] == "cursor"


class TestGetConfigEnvVarOverride:
    """AGENT_SKILLS_PATH env var overrides everything."""

    def test_env_var_wins_over_config(self, clean_env, tmp_path):
        clean_env.setenv("AGENT_SKILLS_PATH", "/override/skills")
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert cfg["skills_path"] == "/override/skills"

    def test_env_var_wins_over_file(self, clean_env, tmp_path):
        clean_env.setenv("AGENT_SKILLS_PATH", "/env/skills")
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (tmp_path / "config.json").write_text(json.dumps({"skills_path": "/file/skills"}))
        fake_script = str(scripts_dir / "generate_mirror.py")
        with mock.patch.object(gm.os.path, "abspath", return_value=fake_script):
            cfg = gm.get_config()
        assert cfg["skills_path"] == "/env/skills"

    def test_agent_provider_env_var(self, clean_env, tmp_path):
        clean_env.setenv("AGENT_PROVIDER", "cursor")
        with mock.patch("os.path.exists", return_value=False):
            cfg = gm.get_config()
        assert cfg.get("provider") == "cursor"


# ===========================================================================
# 8. download_package – subprocess mocked (no real LLM / network calls)
# ===========================================================================

class TestDownloadPackage:
    def _make_fake_tarball(self, tmp_path, pkg_name):
        """Create a minimal .tar.gz archive to satisfy extraction logic."""
        src = tmp_path / "src"
        src.mkdir()
        pkg = src / pkg_name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# fake")
        tb_path = tmp_path / f"{pkg_name}-1.0.0.tar.gz"
        with tarfile.open(str(tb_path), "w:gz") as tar:
            tar.add(str(src / pkg_name), arcname=pkg_name)
        return tb_path

    def test_calls_pip_download(self, tmp_path):
        tb = self._make_fake_tarball(tmp_path, "fakepkg")
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()
        # Simulate pip writing the tarball into dl_dir
        import shutil
        shutil.copy(str(tb), str(dl_dir / tb.name))

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            result = gm.download_package("fakepkg", "1.0.0", str(dl_dir))

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "pip" in cmd
        assert "fakepkg==1.0.0" in cmd

    def test_subprocess_check_true(self, tmp_path):
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()
        tb = self._make_fake_tarball(tmp_path, "fakepkg")
        import shutil
        shutil.copy(str(tb), str(dl_dir / tb.name))

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            gm.download_package("fakepkg", "1.0.0", str(dl_dir))

        _, kwargs = mock_run.call_args
        assert kwargs.get("check") is True


# ===========================================================================
# 9. End-to-end (subprocess mocked – no network, no LLM)
# ===========================================================================

class TestEndToEnd:
    """Simulate a full run without touching the network."""

    def test_full_pipeline(self, tmp_path, tmp_pkg):
        """build_grep_map → generate_skill produces a readable skill."""
        out_dir = tmp_path / "skills"
        out_dir.mkdir()

        grep_map = gm.build_grep_map(str(tmp_pkg))
        assert grep_map, "grep map must not be empty"

        skill_dir = Path(gm.generate_skill("mypkg", "1.2.3", grep_map, str(tmp_pkg), str(out_dir)))

        skill_md = skill_dir / "SKILL.md"
        assert skill_md.is_file()
        content = skill_md.read_text()

        # All parsed symbols appear in the skill map
        for entry in grep_map:
            assert entry["name"] in content

    @pytest.mark.parametrize("provider", list(gm.PROVIDER_DEFAULTS.keys()))
    def test_provider_defaults_table_complete(self, provider):
        """Every documented provider has a non-empty default path."""
        path = gm.PROVIDER_DEFAULTS[provider]
        assert path and path.startswith("~"), f"Provider {provider!r} default path looks wrong: {path!r}"

    @pytest.mark.parametrize("provider", list(gm.PROVIDER_DEFAULTS.keys()))
    def test_path_based_detection_all_providers(self, clean_env, tmp_path, provider):
        """Path-based detection works for every known provider."""
        fake_skills = tmp_path / "skills"
        fake_skills.mkdir()
        script = fake_skills / "obscure-package-master" / "scripts" / "generate_mirror.py"
        defaults = {provider: str(fake_skills)}
        assert gm.detect_provider(provider_defaults=defaults, script_path=str(script)) == provider

    def test_api_keys_never_influence_detection(self, clean_env):
        """Setting provider API key vars must not affect detection results."""
        for var in (
            "ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_GENERATIVEAI_API_KEY", "OPENAI_API_KEY",
            "CODEX_API_KEY", "GITHUB_COPILOT_TOKEN",
        ):
            clean_env.setenv(var, "sk-dummy")
        result = gm.detect_provider(script_path="/tmp/nowhere/scripts/generate_mirror.py")
        assert result is None


# ===========================================================================
# 10. _get_site_packages_candidates
# ===========================================================================

class TestGetSitePackagesCandidates:
    def test_returns_list(self):
        result = gm._get_site_packages_candidates()
        assert isinstance(result, list)

    def test_includes_current_env(self):
        """At least one entry from the live Python environment is present."""
        import site as _site
        result = gm._get_site_packages_candidates()
        try:
            current = _site.getsitepackages()
        except AttributeError:
            pytest.skip("getsitepackages() not available in this build")
        assert any(sp in result for sp in current)

    def test_includes_dotenv_venv(self, tmp_path, monkeypatch):
        """A .venv/lib/pythonX.Y/site-packages directory (Unix layout) is discovered."""
        sp = tmp_path / ".venv" / "lib" / "python3.9" / "site-packages"
        sp.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        result = gm._get_site_packages_candidates()
        assert str(sp) in result

    def test_includes_dotenv_venv_windows_layout(self, tmp_path, monkeypatch):
        """A .venv/Lib/site-packages directory (Windows layout) is discovered."""
        sp = tmp_path / ".venv" / "Lib" / "site-packages"
        sp.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        result = gm._get_site_packages_candidates()
        assert str(sp) in result

    def test_venv_without_lib_ignored(self, tmp_path, monkeypatch):
        """A venv directory that has no lib/ sub-dir causes no error."""
        (tmp_path / "venv").mkdir()
        monkeypatch.chdir(tmp_path)
        result = gm._get_site_packages_candidates()
        assert isinstance(result, list)


# ===========================================================================
# 11. find_local_package
# ===========================================================================

class TestFindLocalPackage:
    def _make_fake_site_packages(self, base, pkg_name, version):
        """Create a minimal site-packages tree with dist-info and package dir."""
        norm = pkg_name.lower().replace("-", "_")
        sp = base / "site-packages"
        sp.mkdir(parents=True, exist_ok=True)
        # dist-info directory signals that the package is installed
        (sp / f"{norm}-{version}.dist-info").mkdir()
        # Package source directory
        pkg_dir = sp / norm
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# fake installed package")
        return sp

    def test_finds_package_in_current_env(self, tmp_path, monkeypatch):
        sp = self._make_fake_site_packages(tmp_path, "mypkg", "1.0.0")
        monkeypatch.setattr(gm, "_get_site_packages_candidates", lambda: [str(sp)])
        result = gm.find_local_package("mypkg", "1.0.0")
        assert result is not None
        assert result.endswith("mypkg")

    def test_returns_none_when_version_mismatch(self, tmp_path, monkeypatch):
        sp = self._make_fake_site_packages(tmp_path, "mypkg", "2.0.0")
        monkeypatch.setattr(gm, "_get_site_packages_candidates", lambda: [str(sp)])
        result = gm.find_local_package("mypkg", "1.0.0")
        assert result is None

    def test_returns_none_when_package_absent(self, tmp_path, monkeypatch):
        sp = tmp_path / "site-packages"
        sp.mkdir()
        monkeypatch.setattr(gm, "_get_site_packages_candidates", lambda: [str(sp)])
        result = gm.find_local_package("mypkg", "1.0.0")
        assert result is None

    def test_finds_package_with_hyphen_in_name(self, tmp_path, monkeypatch):
        """Package names with hyphens are normalised to underscores."""
        sp = self._make_fake_site_packages(tmp_path, "my-pkg", "3.1.4")
        monkeypatch.setattr(gm, "_get_site_packages_candidates", lambda: [str(sp)])
        result = gm.find_local_package("my-pkg", "3.1.4")
        assert result is not None

    def test_scans_multiple_candidates(self, tmp_path, monkeypatch):
        """Returns a match from the second candidate when the first has no match."""
        sp1 = tmp_path / "sp1"
        sp1.mkdir()
        sp2_base = tmp_path / "sp2_base"
        sp2 = self._make_fake_site_packages(sp2_base, "mypkg", "1.0.0")
        monkeypatch.setattr(gm, "_get_site_packages_candidates", lambda: [str(sp1), str(sp2)])
        result = gm.find_local_package("mypkg", "1.0.0")
        assert result is not None

    def test_finds_package_in_nearby_venv(self, tmp_path, monkeypatch):
        """Package installed in a .venv in CWD is discovered on both Unix and Windows layouts."""
        # Create both layouts so the test is platform-independent
        unix_sp_base = tmp_path / ".venv" / "lib" / "python3.9"
        self._make_fake_site_packages(unix_sp_base, "mypkg", "0.5.0")
        win_sp_base = tmp_path / ".venv" / "Lib"
        self._make_fake_site_packages(win_sp_base, "mypkg", "0.5.0")
        monkeypatch.chdir(tmp_path)
        # Don't patch _get_site_packages_candidates so the real function runs
        result = gm.find_local_package("mypkg", "0.5.0")
        assert result is not None


# ===========================================================================
# 12. generate_skill with use_symlinks=True
# ===========================================================================

class TestGenerateSkillSymlinks:
    @requires_symlinks
    def test_symlinks_created_instead_of_copies(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
        ))
        ref_init = skill_dir / "references" / "__init__.py"
        assert ref_init.is_symlink(), "Expected a symlink, not a regular file"

    @requires_symlinks
    def test_symlinks_point_to_original_source(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
        ))
        ref_init = skill_dir / "references" / "__init__.py"
        assert ref_init.resolve() == (tmp_pkg / "__init__.py").resolve()

    @requires_symlinks
    def test_symlinks_are_relative(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
        ))
        ref_init = skill_dir / "references" / "__init__.py"
        target = os.readlink(str(ref_init))
        assert not os.path.isabs(target), "Symlink target should be relative for portability"

    def test_symlinks_readable(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
        ))
        ref_init = skill_dir / "references" / "__init__.py"
        assert ref_init.read_text() == (tmp_pkg / "__init__.py").read_text()

    def test_copy_mode_still_produces_regular_file(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=False
        ))
        ref_init = skill_dir / "references" / "__init__.py"
        assert ref_init.is_file() and not ref_init.is_symlink()

    def test_skill_md_present_when_symlinks(self, tmp_path, tmp_pkg):
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        skill_dir = Path(gm.generate_skill(
            "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
        ))
        assert (skill_dir / "SKILL.md").is_file()

    def test_falls_back_to_copy_when_symlink_fails(self, tmp_path, tmp_pkg):
        """When symlink creation raises OSError, a regular file copy is used instead."""
        out_dir = tmp_path / "skills"
        out_dir.mkdir()
        grep_map = gm.build_grep_map(str(tmp_pkg))
        with mock.patch("os.symlink", side_effect=OSError("symlinks not permitted")):
            skill_dir = Path(gm.generate_skill(
                "mypkg", "1.0.0", grep_map, str(tmp_pkg), str(out_dir), use_symlinks=True
            ))
        ref_init = skill_dir / "references" / "__init__.py"
        assert ref_init.is_file()
        assert not ref_init.is_symlink()
        assert ref_init.read_text() == (tmp_pkg / "__init__.py").read_text()


# ===========================================================================
# 13. Safe archive extraction helpers
# ===========================================================================

class TestSafeExtractTar:
    def _make_legit_tar(self, tmp_path):
        src = tmp_path / "pkg" / "__init__.py"
        src.parent.mkdir(parents=True)
        src.write_text("# safe")
        tb = tmp_path / "legit.tar.gz"
        with tarfile.open(str(tb), "w:gz") as tar:
            tar.add(str(src.parent), arcname="pkg")
        return tb

    def test_extracts_safe_members(self, tmp_path):
        tb = self._make_legit_tar(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        with tarfile.open(str(tb)) as tar:
            gm._safe_extract_tar(tar, str(out))
        assert (out / "pkg" / "__init__.py").is_file()

    def test_skips_path_traversal_members(self, tmp_path, capsys):
        import io
        tb_path = tmp_path / "evil.tar.gz"
        with tarfile.open(str(tb_path), "w:gz") as tar:
            info = tarfile.TarInfo(name="../../evil.txt")
            content = b"evil"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        out = tmp_path / "out"
        out.mkdir()
        with tarfile.open(str(tb_path)) as tar:
            gm._safe_extract_tar(tar, str(out))
        assert not (tmp_path / "evil.txt").exists()
        assert "Skipping" in capsys.readouterr().out

    def test_skips_symlink_members(self, tmp_path, capsys):
        tb_path = tmp_path / "sym.tar.gz"
        with tarfile.open(str(tb_path), "w:gz") as tar:
            info = tarfile.TarInfo(name="link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        out = tmp_path / "out"
        out.mkdir()
        with tarfile.open(str(tb_path)) as tar:
            gm._safe_extract_tar(tar, str(out))
        assert not (out / "link").exists()
        assert "Skipping" in capsys.readouterr().out

    def test_skips_hardlink_members(self, tmp_path, capsys):
        tb_path = tmp_path / "hard.tar.gz"
        with tarfile.open(str(tb_path), "w:gz") as tar:
            info = tarfile.TarInfo(name="hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        out = tmp_path / "out"
        out.mkdir()
        with tarfile.open(str(tb_path)) as tar:
            gm._safe_extract_tar(tar, str(out))
        assert not (out / "hardlink").exists()
        assert "Skipping" in capsys.readouterr().out


class TestSafeExtractZip:
    def _make_legit_zip(self, tmp_path):
        import zipfile as zf
        zp = tmp_path / "legit.zip"
        with zf.ZipFile(str(zp), "w") as z:
            z.writestr("pkg/__init__.py", "# safe")
        return zp

    def test_extracts_safe_members(self, tmp_path):
        import zipfile as zf
        zp = self._make_legit_zip(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        with zf.ZipFile(str(zp)) as z:
            gm._safe_extract_zip(z, str(out))
        assert (out / "pkg" / "__init__.py").is_file()

    def test_skips_path_traversal_members(self, tmp_path, capsys):
        import zipfile as zf
        zp = tmp_path / "evil.zip"
        with zf.ZipFile(str(zp), "w") as z:
            z.writestr("../../evil.txt", "evil")
        out = tmp_path / "out"
        out.mkdir()
        with zf.ZipFile(str(zp)) as z:
            gm._safe_extract_zip(z, str(out))
        assert not (tmp_path / "evil.txt").exists()
        assert "Skipping" in capsys.readouterr().out


# ===========================================================================
# 14. _resolve_skills_dir
# ===========================================================================

class TestResolveSkillsDir:
    """Unit tests for the output-directory priority logic."""

    def test_explicit_path_wins_over_everything(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path="/explicit/skills",
            local=True,
            config_skills_path="/config/skills",
        )
        assert result == "/explicit/skills"

    def test_explicit_path_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path="~/myskills",
            local=False,
            config_skills_path="/config/skills",
        )
        assert result == os.path.expanduser("~/myskills")

    def test_local_flag_with_provider_returns_cwd_relative_path(self, tmp_path, monkeypatch):
        """--local + known provider → project-local provider hidden directory."""
        monkeypatch.chdir(tmp_path)
        home = os.path.expanduser("~")
        global_path = os.path.expanduser("~/.claude/skills")
        expected = os.path.normpath(str(tmp_path) + global_path[len(home):])
        result = gm._resolve_skills_dir(
            explicit_path=None,
            local=True,
            config_skills_path="/config/skills",
            provider="claude",
            provider_defaults={"claude": "~/.claude/skills"},
        )
        assert result == expected

    def test_local_flag_without_provider_returns_none(self, tmp_path, monkeypatch):
        """--local without a detectable provider returns None (caller must error)."""
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path=None,
            local=True,
            config_skills_path="/config/skills",
        )
        assert result is None

    def test_config_path_used_when_no_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path=None,
            local=False,
            config_skills_path="/config/skills",
        )
        assert result == "/config/skills"

    def test_config_path_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path=None,
            local=False,
            config_skills_path="~/provider/skills",
        )
        assert result == os.path.expanduser("~/provider/skills")

    def test_returns_none_when_nothing_configured(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path=None,
            local=False,
            config_skills_path=None,
        )
        assert result is None

    def test_explicit_path_wins_over_none_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = gm._resolve_skills_dir(
            explicit_path="/explicit/path",
            local=False,
            config_skills_path=None,
        )
        assert result == "/explicit/path"
