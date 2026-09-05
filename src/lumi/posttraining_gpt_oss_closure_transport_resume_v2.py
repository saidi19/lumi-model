"""Transport-resilient, zero-regeneration resume for closure generation v0.1."""

from __future__ import annotations

import http.client
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_closure_generation_v1 as closure
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256

SUCCESSOR_SCHEMA = "lumi.sft-gpt-oss-closure-transport-resume.v0.2"
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 3
HEALTH_URL = f"http://127.0.0.1:{base.LOCAL_PORT}/health"


class RecoveredTransportAttemptError(Exception):
    """A counted inference attempt failed, but transport is ready for the next attempt."""


def _is_transport_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            http.client.RemoteDisconnected,
            ConnectionError,
            TimeoutError,
            urllib.error.URLError,
        ),
    ):
        return True
    return isinstance(error, RuntimeError) and "local tunneled vLLM request failed" in str(
        error
    )


class ResilientTunnel:
    """Own one pinned tunnel and recover it without replaying an inference call."""

    def __init__(
        self,
        *,
        ssh_key: Path,
        known_hosts: Path,
        artifact: Path,
        authorization_path: Path,
    ) -> None:
        self.ssh_key = ssh_key
        self.known_hosts = known_hosts
        self.artifact = artifact
        self.authorization_path = authorization_path
        self.process: subprocess.Popen[str] | None = None
        self.consecutive_transport_failures = 0
        self.recoveries = 0

    def start(self) -> None:
        if base._port_open():  # noqa: SLF001
            raise RuntimeError("local port 18080 is already occupied")
        process = subprocess.Popen(
            base._ssh_tunnel_command(self.ssh_key, self.known_hosts),  # noqa: SLF001
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.process = process
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not base._port_open():  # noqa: SLF001
            if process.poll() is not None:
                stderr = "" if process.stderr is None else process.stderr.read()
                raise RuntimeError(f"pinned SSH tunnel exited: {stderr.strip()}")
            time.sleep(0.25)
        if not base._port_open():  # noqa: SLF001
            self.stop()
            raise RuntimeError("pinned SSH tunnel did not open localhost:18080")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def health_ok(self) -> bool:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=10) as response:  # noqa: S310
                return int(response.status) < 400
        except (OSError, TimeoutError, urllib.error.URLError):
            return False

    def recover(self, error: BaseException) -> None:
        """Recover transport before the caller consumes its next authorized attempt."""
        self.consecutive_transport_failures += 1
        self.recoveries += 1
        self.stop()
        self.start()
        action = "tunnel_restarted"
        if not self.health_ok():
            self.stop()
            closure._ssh(  # noqa: SLF001
                self.ssh_key,
                self.known_hosts,
                f"sudo docker restart --time 30 {closure.CONTAINER_ID} >/dev/null; "
                "for i in $(seq 1 240); do "
                "curl -fsS --max-time 3 http://127.0.0.1:18080/health >/dev/null && exit 0; "
                "sleep 2; done; exit 1",
                timeout=540,
            )
            closure.prepare_runtime(
                artifact=self.artifact,
                authorization_path=self.authorization_path,
                ssh_key=self.ssh_key,
                known_hosts=self.known_hosts,
            )
            self.start()
            action = "bound_container_and_tunnel_restarted"
        if not self.health_ok():
            raise RuntimeError("local tunneled vLLM request failed after bounded recovery")
        print(
            "transport_recovery="
            f"{self.recoveries} consecutive={self.consecutive_transport_failures} "
            f"error={type(error).__name__} action={action} health=pass",
            flush=True,
        )

    def mark_success(self) -> None:
        self.consecutive_transport_failures = 0


_ACTIVE_TUNNEL: ResilientTunnel | None = None
_ORIGINAL_POST: Callable[..., dict[str, Any]] | None = None


def _recovering_post(
    payload: Mapping[str, Any], *, timeout: int = 300
) -> dict[str, Any]:
    if _ORIGINAL_POST is None or _ACTIVE_TUNNEL is None:
        raise RuntimeError("transport recovery wrapper is not active")
    try:
        response = _ORIGINAL_POST(payload, timeout=timeout)
    except Exception as error:
        if not _is_transport_error(error):
            raise
        _ACTIVE_TUNNEL.recover(error)
        if (
            _ACTIVE_TUNNEL.consecutive_transport_failures
            >= MAX_CONSECUTIVE_TRANSPORT_FAILURES
        ):
            raise RuntimeError(
                "local tunneled vLLM request failed after three consecutive recovered attempts"
            ) from error
        raise RecoveredTransportAttemptError(
            f"{type(error).__name__}:transport_recovered_for_next_bound_attempt"
        ) from error
    _ACTIVE_TUNNEL.mark_success()
    return response


def _validate_resume_prefix(plan: Mapping[str, Any], root: Path) -> None:
    successor = plan.get("transport_resume")
    resume = plan.get("resume_prefix")
    if not isinstance(successor, Mapping) or not isinstance(resume, Mapping):
        raise ValueError("transport-resume binding missing")
    if successor.get("schema_version") != SUCCESSOR_SCHEMA:
        raise ValueError("transport-resume schema changed")
    expected = successor.get("predecessor")
    if not isinstance(expected, Mapping):
        raise ValueError("transport-resume predecessor binding missing")
    report_path = (root / str(expected.get("report_path"))).resolve()
    report = base._load_json(report_path)  # noqa: SLF001
    if (
        file_sha256(report_path) != expected.get("report_file_sha256")
        or report.get("status") != "FAIL_CLOSED_latest_checkpoint_retained"
        or report.get("early_stop_reason") != "remote_localhost_transport_failure"
    ):
        raise ValueError("transport-resume predecessor report changed")
    paths = {
        "candidate": (root / str(resume.get("candidate_path"))).resolve(),
        "telemetry": (root / str(resume.get("telemetry_path"))).resolve(),
        "attempts": (root / str(resume.get("attempts_path"))).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != expected.get(f"{name}_sha256"):
            raise ValueError(f"transport-resume {name} binding changed")
    rows = base._read_jsonl(paths["candidate"])  # noqa: SLF001
    telemetry = base._read_jsonl(paths["telemetry"])  # noqa: SLF001
    attempts = base._read_jsonl(paths["attempts"])  # noqa: SLF001
    observed = {
        "accepted_outputs": len(rows),
        "inference_calls": len(attempts),
        "retry_calls": sum(item.get("attempt_number") == 2 for item in attempts),
        "telemetry_rows": len(telemetry),
    }
    if observed != expected.get("counts"):
        raise ValueError("transport-resume predecessor counts changed")
    if {row.get("record_id") for row in rows} != {
        item.get("record_id") for item in telemetry
    }:
        raise ValueError("transport-resume predecessor membership changed")


def run_transport_resume(
    *,
    plan_path: Path,
    authorization_path: Path,
    repository_root: Path,
    artifact_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    """Resume v0.1 with reconnectable transport and all original combined caps."""
    global _ACTIVE_TUNNEL, _ORIGINAL_POST
    root = repository_root.resolve()
    plan = closure._load_plan(plan_path.resolve(), root)  # noqa: SLF001
    _validate_resume_prefix(plan, root)
    original_tunnel = base.pinned_ssh_tunnel
    original_post = gpt._post_chat_completion  # noqa: SLF001

    @contextmanager
    def resilient_context(key: Path, hosts: Path) -> Iterator[None]:
        global _ACTIVE_TUNNEL
        manager = ResilientTunnel(
            ssh_key=key,
            known_hosts=hosts,
            artifact=artifact_root.resolve(),
            authorization_path=authorization_path.resolve(),
        )
        _ACTIVE_TUNNEL = manager
        manager.start()
        try:
            yield
        finally:
            manager.stop()
            _ACTIVE_TUNNEL = None

    _ORIGINAL_POST = original_post
    base.pinned_ssh_tunnel = resilient_context
    gpt._post_chat_completion = _recovering_post  # type: ignore[assignment]  # noqa: SLF001
    try:
        return closure.run_closure_generation(
            plan_path=plan_path,
            authorization_path=authorization_path,
            repository_root=root,
            artifact_root=artifact_root,
            ssh_key=ssh_key,
            known_hosts=known_hosts,
        )
    finally:
        gpt._post_chat_completion = original_post  # type: ignore[assignment]  # noqa: SLF001
        base.pinned_ssh_tunnel = original_tunnel
        _ORIGINAL_POST = None
