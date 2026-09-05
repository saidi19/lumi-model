import http.client

import pytest

from lumi import posttraining_gpt_oss_closure_transport_resume_v2 as transport


class FakeTunnel:
    def __init__(self) -> None:
        self.consecutive_transport_failures = 0
        self.recoveries = 0

    def recover(self, error: BaseException) -> None:
        del error
        self.consecutive_transport_failures += 1
        self.recoveries += 1

    def mark_success(self) -> None:
        self.consecutive_transport_failures = 0


def test_remote_disconnect_is_recovered_without_hidden_replay(
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


def test_success_resets_consecutive_failure_count(monkeypatch: pytest.MonkeyPatch) -> None:
    tunnel = FakeTunnel()
    tunnel.consecutive_transport_failures = 2
    monkeypatch.setattr(
        transport,
        "_ORIGINAL_POST",
        lambda payload, *, timeout: {"payload": payload, "timeout": timeout},
    )
    monkeypatch.setattr(transport, "_ACTIVE_TUNNEL", tunnel)
    response = transport._recovering_post({"seed": 2}, timeout=7)
    assert response == {"payload": {"seed": 2}, "timeout": 7}
    assert tunnel.consecutive_transport_failures == 0


def test_three_consecutive_transport_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tunnel = FakeTunnel()
    tunnel.consecutive_transport_failures = 2
    monkeypatch.setattr(
        transport,
        "_ORIGINAL_POST",
        lambda payload, *, timeout: (_ for _ in ()).throw(
            RuntimeError("local tunneled vLLM request failed: connection refused")
        ),
    )
    monkeypatch.setattr(transport, "_ACTIVE_TUNNEL", tunnel)
    with pytest.raises(RuntimeError, match="three consecutive"):
        transport._recovering_post({"seed": 3})
