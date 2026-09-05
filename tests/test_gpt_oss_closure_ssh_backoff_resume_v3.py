import http.client

import pytest

from lumi import posttraining_gpt_oss_closure_ssh_backoff_resume_v3 as transport


class FakeTunnel:
    def __init__(self) -> None:
        self.consecutive_transport_attempts = 0
        self.recoveries = 0

    def recover(self, error: BaseException) -> None:
        del error
        self.consecutive_transport_attempts += 1
        self.recoveries += 1

    def mark_success(self) -> None:
        self.consecutive_transport_attempts = 0


def test_remote_disconnect_has_no_hidden_inference_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def post(payload: object, *, timeout: int) -> dict[str, object]:
        del payload, timeout
        nonlocal calls
        calls += 1
        raise http.client.RemoteDisconnected("peer closed")

    tunnel = FakeTunnel()
    monkeypatch.setattr(transport, "_ORIGINAL_POST", post)
    monkeypatch.setattr(transport, "_ACTIVE_TUNNEL", tunnel)
    with pytest.raises(transport.RecoveredTransportAttemptError):
        transport._recovering_post({"seed": 1})
    assert calls == 1
    assert tunnel.recoveries == 1


def test_success_resets_transport_incident_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tunnel = FakeTunnel()
    tunnel.consecutive_transport_attempts = 19
    monkeypatch.setattr(
        transport,
        "_ORIGINAL_POST",
        lambda payload, *, timeout: {"payload": payload, "timeout": timeout},
    )
    monkeypatch.setattr(transport, "_ACTIVE_TUNNEL", tunnel)
    assert transport._recovering_post({"seed": 2}, timeout=7)["timeout"] == 7
    assert tunnel.consecutive_transport_attempts == 0


def test_lifecycle_backoff_does_not_consume_inference() -> None:
    attempts = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise RuntimeError("connection reset")
        return "ready"

    assert transport._retry_lifecycle(
        operation,
        label="test",
        delays=(0, 1, 2, 3),
        sleep=sleeps.append,
    ) == "ready"
    assert attempts == 4
    assert sleeps == [1, 2, 3]


def test_twenty_consecutive_transport_attempts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tunnel = FakeTunnel()
    tunnel.consecutive_transport_attempts = 19
    monkeypatch.setattr(
        transport,
        "_ORIGINAL_POST",
        lambda payload, *, timeout: (_ for _ in ()).throw(
            RuntimeError("local tunneled vLLM request failed: connection refused")
        ),
    )
    monkeypatch.setattr(transport, "_ACTIVE_TUNNEL", tunnel)
    with pytest.raises(RuntimeError, match="twenty consecutive"):
        transport._recovering_post({"seed": 3})


def test_tunnel_command_has_explicit_liveness_controls() -> None:
    command = transport._tunnel_command(
        transport.Path("/tmp/key"), transport.Path("/tmp/known-hosts")
    )
    joined = " ".join(command)
    assert "ServerAliveInterval=10" in joined
    assert "ServerAliveCountMax=6" in joined
    assert "TCPKeepAlive=yes" in joined
    assert "ConnectionAttempts=3" in joined

