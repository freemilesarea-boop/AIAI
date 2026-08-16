"""The single-worker guard and the launchd agent generated for it.

Two things are worth testing here and one is not. The lock's behaviour
is: it excludes a second worker on this host, the kernel releases it when
the holder dies however it dies, and it is scoped to the machine so a
future multi-host deployment stays legal. The generated agent's contents
are checkable too — above all that nobody's home directory is baked into
a file the repository ships.

What is not tested is launchd itself. Whether Apple honours KeepAlive is
not this repository's property to assert.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from luber_generation_worker.singleton import (
    EXIT_ALREADY_RUNNING,
    SingleWorkerLock,
    WorkerAlreadyRunningError,
    log_dir,
    runtime_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_TOOL = REPO_ROOT / "scripts" / "development" / "luber_runtime.py"


def load_runtime_tool():
    """Import the control script by path; it is a script, not a package."""
    spec = importlib.util.spec_from_file_location("luber_runtime", RUNTIME_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSingleWorkerLock:
    def test_the_first_holder_gets_it(self, tmp_path):
        lock = SingleWorkerLock(tmp_path / "w.lock")
        lock.acquire()
        try:
            assert lock.read_holder_pid() == os.getpid()
        finally:
            lock.release()

    def test_a_second_holder_is_refused(self, tmp_path):
        path = tmp_path / "w.lock"
        first = SingleWorkerLock(path)
        first.acquire()
        try:
            with pytest.raises(WorkerAlreadyRunningError) as caught:
                SingleWorkerLock(path).acquire()
            # The operator needs to know who to go and stop.
            assert caught.value.holder_pid == os.getpid()
        finally:
            first.release()

    def test_release_lets_the_next_one_in(self, tmp_path):
        path = tmp_path / "w.lock"
        first = SingleWorkerLock(path)
        first.acquire()
        first.release()
        second = SingleWorkerLock(path)
        second.acquire()  # must not raise
        second.release()

    def test_a_killed_holder_does_not_leave_it_stuck(self, tmp_path):
        """The property a PID file cannot offer.

        SIGKILL gives the holder no chance to clean up. The kernel drops
        the lock anyway, so recovery from a crash needs no operator step
        and there is no stale state to detect.
        """
        path = tmp_path / "w.lock"
        script = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(REPO_ROOT / "services" / "generation-worker" / "src")!r})
            from luber_generation_worker.singleton import SingleWorkerLock
            lock = SingleWorkerLock({str(path)!r})
            lock.acquire()
            print("held", flush=True)
            time.sleep(60)
        """)
        child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "held"
            with pytest.raises(WorkerAlreadyRunningError):
                SingleWorkerLock(path).acquire()
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:  # pragma: no cover - only on a hung child
                child.kill()

        reclaimed = SingleWorkerLock(path)
        reclaimed.acquire()  # must not raise
        reclaimed.release()

    def test_separate_paths_do_not_exclude_each_other(self, tmp_path):
        """Why this cannot forbid a future multi-host deployment.

        The lock is a file on one machine. Two hosts have two files, so
        two workers on two machines — a legitimate topology — are
        unaffected. Exactly-once processing across hosts is ARQ's job.
        """
        a = SingleWorkerLock(tmp_path / "host-a.lock")
        b = SingleWorkerLock(tmp_path / "host-b.lock")
        a.acquire()
        b.acquire()  # must not raise
        a.release()
        b.release()

    def test_the_refusal_names_a_distinct_exit_code(self):
        """So a supervisor's log separates "already running" from a fault."""
        assert EXIT_ALREADY_RUNNING == 3
        assert EXIT_ALREADY_RUNNING != 0
        assert EXIT_ALREADY_RUNNING != 1


class TestRuntimePaths:
    def test_runtime_and_log_paths_are_resolved_not_hardcoded(self):
        """The paths contain a home directory; the source must not."""
        for path in (runtime_dir(), log_dir()):
            assert path.is_absolute()
        source = Path(runtime_dir.__code__.co_filename).read_text() + RUNTIME_TOOL.read_text()
        assert "/Users/" not in source

    def test_the_log_directory_is_overridable(self, monkeypatch, tmp_path):
        """Deployment sets this; the default is only a development default."""
        monkeypatch.setenv("LUBER_LOG_DIR", str(tmp_path / "logs"))
        assert log_dir() == tmp_path / "logs"

    def test_xdg_runtime_dir_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        assert runtime_dir() == tmp_path / "run"


class TestLaunchdAgent:
    def test_the_committed_template_contains_no_personal_path(self):
        template = (
            REPO_ROOT / "deploy" / "launchd" / "com.luber.generation-worker.plist.template"
        ).read_text()
        assert "/Users/" not in template
        assert str(Path.home()) not in template
        # The machine-specific values must all still be placeholders.
        for placeholder in ("@ARQ@", "@REPO@", "@LOG_DIR@", "@PATH@", "@LABEL@"):
            assert placeholder in template

    def test_rendering_fills_every_placeholder(self):
        rendered = load_runtime_tool().render_plist()
        # The template's own explanatory comment mentions @PLACEHOLDER@,
        # so check the real placeholders rather than every "@".
        for placeholder in ("@LABEL@", "@ARQ@", "@REPO@", "@LOG_DIR@", "@PATH@"):
            assert placeholder not in rendered, f"{placeholder} was not filled in"
        assert "com.luber.generation-worker" in rendered

    def test_the_rendered_agent_runs_the_worker_from_the_repository(self):
        module = load_runtime_tool()
        rendered = module.render_plist()
        assert "luber_generation_worker.worker.WorkerSettings" in rendered
        assert str(module.REPO_ROOT) in rendered
        assert str(module.arq_path()) in rendered

    def test_restart_is_throttled_and_not_unconditional(self):
        """A clean exit is a deliberate stop, not something to undo."""
        rendered = load_runtime_tool().render_plist()
        assert "<key>ThrottleInterval</key>" in rendered
        assert "<key>SuccessfulExit</key>" in rendered
        # KeepAlive as a bare <true/> would make `stop` impossible.
        assert "<key>KeepAlive</key>\n    <true/>" not in rendered

    def test_logs_go_to_the_resolved_log_directory(self):
        module = load_runtime_tool()
        rendered = module.render_plist()
        assert f"{log_dir()}/generation-worker.out.log" in rendered
        assert f"{log_dir()}/generation-worker.err.log" in rendered

    def test_the_agent_is_installed_under_the_user_launchagents(self):
        target = load_runtime_tool().installed_plist_path()
        assert target.parent == Path.home() / "Library" / "LaunchAgents"
        assert target.name == "com.luber.generation-worker.plist"

    def test_the_generated_agent_is_valid_property_list(self):
        rendered = load_runtime_tool().render_plist()
        result = subprocess.run(
            ["plutil", "-lint", "-"], input=rendered, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestSafeProcessDiscovery:
    def test_an_unrelated_pid_is_never_reported_as_the_worker(self, monkeypatch, tmp_path):
        """The guard against signalling something that merely reused a PID.

        The lock file names a PID; this test puts a real but unrelated
        process's id in it and requires the tool to reject it.
        """
        module = load_runtime_tool()
        lock_path = tmp_path / "w.lock"
        innocent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            lock_path.write_text(f"{innocent.pid}\n")
            monkeypatch.setattr(
                module, "SingleWorkerLock", lambda *a, **k: SingleWorkerLock(lock_path)
            )
            assert module.running_worker_pid() is None
        finally:
            innocent.kill()
            innocent.wait(timeout=10)

    def test_a_dead_pid_is_not_reported(self, monkeypatch, tmp_path):
        module = load_runtime_tool()
        lock_path = tmp_path / "w.lock"
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=10)
        lock_path.write_text(f"{dead.pid}\n")
        monkeypatch.setattr(module, "SingleWorkerLock", lambda *a, **k: SingleWorkerLock(lock_path))
        assert module.running_worker_pid() is None

    def test_an_empty_lock_file_is_not_a_worker(self, monkeypatch, tmp_path):
        module = load_runtime_tool()
        lock_path = tmp_path / "w.lock"
        lock_path.write_text("")
        monkeypatch.setattr(module, "SingleWorkerLock", lambda *a, **k: SingleWorkerLock(lock_path))
        assert module.running_worker_pid() is None

    def test_stopping_uses_sigterm_not_sigkill(self):
        """SIGKILL would abandon a running generation instead of re-queueing it."""
        source = RUNTIME_TOOL.read_text()
        assert "os.kill(pid, signal.SIGTERM)" in source
        assert "signal.SIGKILL" not in source

    def test_the_tool_never_pattern_kills(self):
        """Checked against what the tool *executes*, not what it says.

        The docstring names ``pkill -f python`` as the thing being
        avoided, so a raw text search would match the explanation. Only
        executed commands count.
        """
        module = load_runtime_tool()
        source = RUNTIME_TOOL.read_text()
        executed = [
            line
            for line in source.splitlines()
            if ("subprocess.run" in line or "subprocess.Popen" in line or "os.system" in line)
        ]
        for line in executed:
            for forbidden in ("pkill", "killall"):
                assert forbidden not in line, f"{forbidden} executed: {line.strip()}"
        # And the only signal it ever sends is the graceful one.
        assert "SIGKILL" not in module.cmd_stop.__code__.co_names
