"""SSH-backoff, zero-regeneration resume for closure generation v0.1."""

from __future__ import annotations

import http.client
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from . import posttraining_gpt_oss_closure_generation_v1 as closure
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256

SUCCESSOR_SCHEMA = "lumi.sft-gpt-oss-closure-ssh-backoff-resume.v0.3"
HEALTH_URL = f"http://127.0.0.1:{base.LOCAL_PORT}/health"
CONTROL_BACKOFF_SECONDS = (0, 5, 10, 20, 30, 45, 60, 60, 60, 60, 60, 60)
LIFECYCLE_BACKOFF_SECONDS = (0, 5, 10, 20, 30, 45, 60, 60)
MAX_CONSECUTIVE_TRANSPORT_ATTEMPTS = 20

T = TypeVar("T")


class RecoveredTransportAttemptError(Exception):
    """A counted model attempt failed, while SSH recovered before the next attempt."""


def _is_model_transport_error(error: BaseException) -> bool:
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
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        marker in message
        for marker in (
            "local tunneled vllm request failed",
            "pinned ssh tunnel exited",
            "connection reset",
            "broken pipe",
        )
    )


def _retry_lifecycle(
    operation: Callable[[], T],
    *,
    label: str,
    delays: Sequence[int] = LIFECYCLE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry SSH lifecycle checks without consuming or replaying inference calls."""
    last_error: BaseException | None = None
    for index, delay in enumerate(delays, 1):
        if delay:
            sleep(delay)
        try:
            return operation()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = error
            print(
                f"ssh_lifecycle_retry={label} attempt={index}/{len(delays)} "
                f"error={type(error).__name__}",
                flush=True,
            )
    raise RuntimeError(
        f"SSH lifecycle recovery exhausted for {label}:"
        f"{type(last_error).__name__ if last_error else 'unknown'}"
    ) from last_error


def _tunnel_command(ssh_key: Path, known_hosts: Path) -> list[str]:
    """Pinned tunnel with explicit TCP and SSH liveness controls."""
    return [
        "ssh",
        "-i",
        str(ssh_key.resolve()),
        "-o",
        f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "IPQoS=none",
        "-N",
        "-L",
        f"{base.LOCAL_PORT}:127.0.0.1:18080",
        f"{gpt.SERVER['ssh_user']}@{gpt.SERVER['host']}",
    ]


class BackoffTunnel:
    """Own and repair one pinned SSH tunnel without hidden inference replay."""

    def __init__(
        self,
        *,
        ssh_key: Path,
        known_hosts: Path,
        artifact: Path,
        authorization_path: Path,
        prepare_runtime: Callable[..., dict[str, Any]],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ssh_key = ssh_key
        self.known_hosts = known_hosts
        self.artifact = artifact
        self.authorization_path = authorization_path
        self.prepare_runtime = prepare_runtime
        self.sleep = sleep
        self.process: subprocess.Popen[str] | None = None
        self.consecutive_transport_attempts = 0
        self.recoveries = 0

    def _start_once(self) -> None:
        if base._port_open():  # noqa: SLF001
            raise RuntimeError("local port 18080 is already occupied")
        process = subprocess.Popen(
            _tunnel_command(self.ssh_key, self.known_hosts),
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
            self.sleep(0.25)
        if not base._port_open():  # noqa: SLF001
            raise RuntimeError("pinned SSH tunnel did not open localhost:18080")
        health_deadline = time.monotonic() + 20
        while time.monotonic() < health_deadline:
            if process.poll() is not None:
                stderr = "" if process.stderr is None else process.stderr.read()
                raise RuntimeError(f"pinned SSH tunnel exited: {stderr.strip()}")
            if self.health_ok():
                return
            self.sleep(1)
        raise RuntimeError("pinned SSH tunnel opened but model health did not pass")

    def start_with_backoff(
        self, delays: Sequence[int] = CONTROL_BACKOFF_SECONDS
    ) -> None:
        last_error: BaseException | None = None
        for index, delay in enumerate(delays, 1):
            self.stop()
            if delay:
                self.sleep(delay)
            try:
                self._start_once()
                if index > 1:
                    print(
                        f"ssh_tunnel_ready_after_attempt={index}/{len(delays)}",
                        flush=True,
                    )
                return
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                last_error = error
                print(
                    f"ssh_tunnel_backoff={index}/{len(delays)} "
                    f"delay_seconds={delay} error={type(error).__name__}",
                    flush=True,
                )
        self.stop()
        raise RuntimeError(
            "local tunneled vLLM request failed: SSH control-plane recovery exhausted:"
            f"{type(last_error).__name__ if last_error else 'unknown'}"
        ) from last_error

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
        """Repair SSH fully before returning control to the next bound attempt."""
        self.consecutive_transport_attempts += 1
        self.recoveries += 1
        self.stop()
        try:
            self.start_with_backoff()
            action = "ssh_tunnel_reestablished"
        except RuntimeError:
            self.prepare_runtime(
                artifact=self.artifact,
                authorization_path=self.authorization_path,
                ssh_key=self.ssh_key,
                known_hosts=self.known_hosts,
            )
            self.start_with_backoff()
            action = "bound_runtime_revalidated_and_tunnel_reestablished"
        print(
            f"transport_recovery={self.recoveries} "
            f"consecutive={self.consecutive_transport_attempts} "
            f"error={type(error).__name__} action={action} health=pass",
            flush=True,
        )

    def mark_success(self) -> None:
        self.consecutive_transport_attempts = 0


_ACTIVE_TUNNEL: BackoffTunnel | None = None
_ORIGINAL_POST: Callable[..., dict[str, Any]] | None = None


def _recovering_post(
    payload: Mapping[str, Any], *, timeout: int = 300
) -> dict[str, Any]:
    if _ORIGINAL_POST is None or _ACTIVE_TUNNEL is None:
        raise RuntimeError("SSH-backoff transport wrapper is not active")
    try:
        response = _ORIGINAL_POST(payload, timeout=timeout)
    except Exception as error:
        if not _is_model_transport_error(error):
            raise
        _ACTIVE_TUNNEL.recover(error)
        if (
            _ACTIVE_TUNNEL.consecutive_transport_attempts
            >= MAX_CONSECUTIVE_TRANSPORT_ATTEMPTS
        ):
            raise RuntimeError(
                "local tunneled vLLM request failed after twenty consecutive "
                "transport-only attempts"
            ) from error
        raise RecoveredTransportAttemptError(
            f"{type(error).__name__}:ssh_backoff_recovered_for_next_bound_attempt"
        ) from error
    _ACTIVE_TUNNEL.mark_success()
    return response


def _validate_resume_prefix(plan: Mapping[str, Any], root: Path) -> None:
    successor = plan.get("ssh_backoff_resume")
    resume = plan.get("resume_prefix")
    if not isinstance(successor, Mapping) or not isinstance(resume, Mapping):
        raise ValueError("SSH-backoff resume binding missing")
    if successor.get("schema_version") != SUCCESSOR_SCHEMA:
        raise ValueError("SSH-backoff resume schema changed")
    predecessor = successor.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise ValueError("SSH-backoff predecessor binding missing")
    report_path = (root / str(predecessor.get("report_path"))).resolve()
    report = base._load_json(report_path)  # noqa: SLF001
    if (
        file_sha256(report_path) != predecessor.get("report_file_sha256")
        or report.get("status") != "FAIL_CLOSED_latest_checkpoint_retained"
        or report.get("early_stop_reason") != "remote_localhost_transport_failure"
    ):
        raise ValueError("SSH-backoff predecessor report changed")
    paths = {
        "candidate": (root / str(resume.get("candidate_path"))).resolve(),
        "telemetry": (root / str(resume.get("telemetry_path"))).resolve(),
        "attempts": (root / str(resume.get("attempts_path"))).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != predecessor.get(f"{name}_sha256"):
            raise ValueError(f"SSH-backoff predecessor {name} binding changed")
    rows = base._read_jsonl(paths["candidate"])  # noqa: SLF001
    telemetry = base._read_jsonl(paths["telemetry"])  # noqa: SLF001
    attempts = base._read_jsonl(paths["attempts"])  # noqa: SLF001
    counts = {
        "accepted_outputs": len(rows),
        "inference_calls": len(attempts),
        "retry_calls": sum(item.get("attempt_number") == 2 for item in attempts),
        "telemetry_rows": len(telemetry),
    }
    if counts != predecessor.get("counts"):
        raise ValueError("SSH-backoff predecessor counts changed")
    if {row.get("record_id") for row in rows} != {
        item.get("record_id") for item in telemetry
    }:
        raise ValueError("SSH-backoff predecessor membership changed")


def run_ssh_backoff_resume(
    *,
    plan_path: Path,
    authorization_path: Path,
    repository_root: Path,
    artifact_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    """Resume with SSH control-plane backoff and all original combined caps."""
    global _ACTIVE_TUNNEL, _ORIGINAL_POST
    root = repository_root.resolve()
    plan = closure._load_plan(plan_path.resolve(), root)  # noqa: SLF001
    _validate_resume_prefix(plan, root)

    original_tunnel = base.pinned_ssh_tunnel
    original_stop = base._stop_runtime  # noqa: SLF001
    original_post = gpt._post_chat_completion  # noqa: SLF001
    original_gpu_snapshot = gpt.gpu_snapshot
    original_prepare = closure.prepare_runtime

    def robust_prepare(**kwargs: Any) -> dict[str, Any]:
        return _retry_lifecycle(
            lambda: original_prepare(**kwargs), label="prepare_bound_runtime"
        )

    def robust_gpu_snapshot(key: Path, hosts: Path) -> dict[str, Any]:
        return _retry_lifecycle(
            lambda: original_gpu_snapshot(key, hosts), label="bound_gpu_snapshot"
        )

    def robust_stop(key: Path, hosts: Path) -> None:
        _retry_lifecycle(
            lambda: original_stop(key, hosts), label="stop_bound_runtime"
        )

    @contextmanager
    def resilient_context(key: Path, hosts: Path) -> Iterator[None]:
        global _ACTIVE_TUNNEL
        manager = BackoffTunnel(
            ssh_key=key,
            known_hosts=hosts,
            artifact=artifact_root.resolve(),
            authorization_path=authorization_path.resolve(),
            prepare_runtime=robust_prepare,
        )
        _ACTIVE_TUNNEL = manager
        manager.start_with_backoff()
        try:
            yield
        finally:
            manager.stop()
            _ACTIVE_TUNNEL = None

    _ORIGINAL_POST = original_post
    base.pinned_ssh_tunnel = resilient_context
    base._stop_runtime = robust_stop  # type: ignore[assignment]  # noqa: SLF001
    gpt._post_chat_completion = _recovering_post  # type: ignore[assignment]  # noqa: SLF001
    gpt.gpu_snapshot = robust_gpu_snapshot  # type: ignore[assignment]
    closure.prepare_runtime = robust_prepare  # type: ignore[assignment]
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
        closure.prepare_runtime = original_prepare  # type: ignore[assignment]
        gpt.gpu_snapshot = original_gpu_snapshot  # type: ignore[assignment]
        gpt._post_chat_completion = original_post  # type: ignore[assignment]  # noqa: SLF001
        base._stop_runtime = original_stop  # type: ignore[assignment]  # noqa: SLF001
        base.pinned_ssh_tunnel = original_tunnel
        _ORIGINAL_POST = None

