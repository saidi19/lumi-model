"""Synthetic daemon metadata and inert syscall doubles; no Docker or native execution."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_oci_admission_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
admission = importlib.import_module(PACKAGE + ".posttraining_sft_oci_admission_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
CONTAINER_ID = "a" * 64
PRIVATE_TEXT = "SYNTHETIC-PRIVATE-EXCEPTION-TEXT"
COUNTERS = (
    "verification_attempts",
    "verification_completed",
    "handoff_attempts",
    "handoff_completed",
)


def metadata():
    environment = {
        "LUMI_RELEASE_LOCK_SHA256": admission.LOCK_SHA256,
        "LUMI_RELEASE_BASE_INVENTORY_SHA256": admission.BASE_SHA256,
        "LUMI_RELEASE_SOURCE_INVENTORY_SHA256": admission.SOURCE_SHA256,
        "LUMI_RELEASE_SOURCE_COMMIT": admission.SOURCE_COMMIT,
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "GLOO_SOCKET_IFNAME": "lo",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    config = {
        "User": "10001:10001",
        "WorkingDir": "/opt/lumi-release/source",
        "Entrypoint": [admission.PYTHON, "-I", "-B"],
        "Cmd": [admission.WRAPPER, "--synthetic-only"],
        "Env": [f"{name}={value}" for name, value in environment.items()],
        "Volumes": None,
        "Labels": None,
    }
    image = {
        "Id": admission.IMAGE_ID,
        "Os": "linux",
        "Architecture": "amd64",
        "Config": config,
    }
    container = {
        "Id": CONTAINER_ID,
        "Image": admission.IMAGE_ID,
        "Config": {**json.loads(json.dumps(config)), "Image": admission.IMAGE_ID},
        "Path": admission.PYTHON,
        "Args": ["-I", "-B", admission.WRAPPER, "--synthetic-only"],
        "RestartCount": 0,
        "State": {
            "Status": "created",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 0,
            "ExitCode": 0,
            "StartedAt": "0001-01-01T00:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z",
        },
        "HostConfig": {
            "ReadonlyRootfs": True,
            "Privileged": False,
            "AutoRemove": False,
            "NetworkMode": "none",
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "IpcMode": "private",
            "Tmpfs": dict(admission.TMPFS),
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Memory": 4294967296,
            "MemorySwap": 4294967296,
            "ShmSize": 67108864,
            "PidsLimit": 256,
            "NanoCpus": 2000000000,
            "Ulimits": [
                {"Name": "core", "Soft": 0, "Hard": 0},
                {"Name": "nofile", "Soft": 1024, "Hard": 1024},
            ],
        },
        "Mounts": [],
    }
    return image, container


def encoded(value):
    return admission.canonical([value])


def change(value, path, replacement):
    for component in path[:-1]:
        value = value[component]
    value[path[-1]] = replacement


class BindingTests(unittest.TestCase):
    def test_import_additions_are_stdlib_and_source_only(self):
        roots = {name.split(".", 1)[0] for name in IMPORT_ADDITIONS}
        self.assertFalse(roots & {"torch", "numpy", "transformers", "cuda", "cupy", "jax"})
        self.assertFalse(
            any("fixture" in name or "worker_probe" in name for name in IMPORT_ADDITIONS)
        )

    def test_binding_has_only_exact_accepted_software_metadata(self):
        selected = admission.SoftwareBinding()
        self.assertEqual(
            {field: getattr(selected, field) for field in selected.__slots__},
            {
                "image_id": admission.IMAGE_ID,
                "source_commit": admission.SOURCE_COMMIT,
                "software_lock_sha256": admission.LOCK_SHA256,
                "base_inventory_sha256": admission.BASE_SHA256,
                "source_inventory_sha256": admission.SOURCE_SHA256,
                "acceptance_report_sha256": admission.ACCEPTANCE_SHA256,
            },
        )
        self.assertFalse(hasattr(selected, "__dict__"))
        for name in admission.NONCLAIMS:
            self.assertFalse(hasattr(selected, name))

    def test_binding_fields_cannot_be_mutated(self):
        selected = admission.SoftwareBinding()
        with self.assertRaises(FrozenInstanceError):
            selected.image_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            del selected.source_commit

    def test_identity_is_domain_separated_deterministic_and_metadata_sensitive(self):
        selected = admission.SoftwareBinding()
        expected = hashlib.sha256(
            admission.canonical(
                {
                    "schema": "lumi.sft-accepted-oci-software-identity.v0.1",
                    **{name: getattr(selected, name) for name in selected.__slots__},
                }
            )
        ).hexdigest()
        self.assertEqual(selected.source_identity_sha256(), expected)
        self.assertEqual(admission.SoftwareBinding().source_identity_sha256(), expected)
        self.assertNotEqual(
            expected,
            admission.SoftwareBinding(
                source_commit="b" * 40,
            ).source_identity_sha256(),
        )

    def test_exception_accepts_only_allowlisted_fixed_codes(self):
        for supplied in (PRIVATE_TEXT, {}, 1, None):
            with self.subTest(supplied=type(supplied).__name__):
                error = admission.OciAdmissionError(supplied)
                self.assertEqual(error.code, "internal_failed")
                self.assertEqual(str(error), "internal_failed")
        self.assertEqual(admission.OciAdmissionError("image_rejected").args, ("image_rejected",))

    def test_public_verifier_cannot_receive_a_caller_selected_image_or_receipt(self):
        self.assertEqual(
            tuple(inspect.signature(admission.verify_created_oci).parameters),
            ("container_id", "deadline"),
        )
        self.assertEqual(
            tuple(inspect.signature(admission.VerifiedOciHandoff).parameters),
            ("container_id", "expected_parent_pid", "deadline", "continuation"),
        )


class CreatedOciTests(unittest.TestCase):
    def setUp(self):
        self.image, self.container = metadata()
        self.inspector = Mock(side_effect=self.inspect)
        self.clock = types.SimpleNamespace(monotonic=Mock(return_value=100.0))
        for name, value in (
            ("_runtime", Mock()),
            ("_inspect", self.inspector),
            ("time", self.clock),
        ):
            selected = patch.object(admission, name, value)
            selected.start()
            self.addCleanup(selected.stop)

    def inspect(self, kind, identifier, deadline):
        self.assertEqual(deadline, 110.0)
        self.assertEqual(identifier, CONTAINER_ID if kind == "container" else admission.IMAGE_ID)
        return encoded(self.container if kind == "container" else self.image)

    def verify(self, **overrides):
        return admission.verify_created_oci(
            overrides.pop("container_id", CONTAINER_ID),
            deadline=overrides.pop("deadline", 110.0),
            **overrides,
        )

    def reject(self, code):
        with self.assertRaises(admission.OciAdmissionError) as caught:
            self.verify()
        error = caught.exception
        self.assertEqual(error.code, code)
        self.assertEqual(error.args, (code,))
        self.assertNotIn(PRIVATE_TEXT, str(error))
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def test_matching_never_started_container_returns_immutable_binding(self):
        selected = self.verify()
        self.assertIs(type(selected), admission.SoftwareBinding)
        self.assertEqual(selected, admission.SoftwareBinding())
        self.assertEqual(
            [call.args[:2] for call in self.inspector.call_args_list],
            [
                ("image", admission.IMAGE_ID),
                ("container", CONTAINER_ID),
            ],
        )

    def test_runtime_is_checked_before_inspection(self):
        admission._runtime.side_effect = admission.OciAdmissionError("runtime_rejected")
        self.reject("runtime_rejected")
        self.inspector.assert_not_called()

    def test_only_full_lowercase_container_id_is_admitted(self):
        for supplied in (
            None,
            True,
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "container-name",
            "--format",
            admission.IMAGE_ID,
        ):
            with self.subTest(supplied=repr(supplied)):
                with self.assertRaises(admission.OciAdmissionError) as caught:
                    self.verify(container_id=supplied)
                self.assertEqual(caught.exception.code, "configuration_rejected")
        self.inspector.assert_not_called()

    def test_deadline_requires_finite_near_future_nonboolean_number(self):
        for supplied in (None, "110", True, 0, 100.0, 116.0, float("inf"), float("nan")):
            with self.subTest(supplied=repr(supplied)):
                with self.assertRaises(admission.OciAdmissionError) as caught:
                    self.verify(deadline=supplied)
                self.assertEqual(caught.exception.code, "configuration_rejected")
        self.inspector.assert_not_called()

    def test_wrong_image_identity_os_or_architecture_is_rejected(self):
        for key, value in (
            ("Id", "sha256:" + "b" * 64),
            ("Os", "windows"),
            ("Architecture", "arm64"),
        ):
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.image[key] = value
                self.reject("image_rejected")

    def test_wrong_container_id_image_path_or_arguments_is_rejected(self):
        for path, value in (
            (("Id",), "b" * 64),
            (("Image",), "sha256:" + "b" * 64),
            (("Config", "Image"), "latest"),
            (("Path",), "/bin/sh"),
            (("Args",), ["-c", "synthetic"]),
        ):
            with self.subTest(path=path):
                self.image, self.container = metadata()
                change(self.container, path, value)
                self.reject("container_rejected")

    def test_started_paused_restarted_dead_or_oom_container_is_rejected(self):
        changes = [
            ("Status", "exited"),
            ("Pid", 1),
            ("Pid", False),
            ("ExitCode", 1),
            ("ExitCode", False),
            ("StartedAt", "2026-01-01T00:00:00Z"),
            ("FinishedAt", "2026-01-01T00:00:00Z"),
        ]
        changes += [
            (key, value)
            for key in ("Running", "Paused", "Restarting", "OOMKilled", "Dead")
            for value in (True, 0)
        ]
        for key, value in changes:
            with self.subTest(field=key, value=value):
                self.image, self.container = metadata()
                self.container["State"][key] = value
                self.reject("container_rejected")
        for value in (1, False):
            self.image, self.container = metadata()
            self.container["RestartCount"] = value
            self.reject("container_rejected")

    def test_container_cannot_override_any_image_launch_configuration(self):
        fields = (
            "Env",
            "Entrypoint",
            "Cmd",
            "User",
            "WorkingDir",
            "Labels",
            "Volumes",
            "Healthcheck",
            "StopSignal",
            "Shell",
            "ExposedPorts",
            "OnBuild",
        )
        for key in fields:
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.container["Config"][key] = ["changed"]
                self.reject("profile_rejected")

    def test_image_and_container_agreement_does_not_waive_fixed_profile(self):
        for key, value in (
            ("User", "0:0"),
            ("WorkingDir", "/tmp"),
            ("Entrypoint", ["/bin/sh"]),
            ("Cmd", ["other"]),
            ("Volumes", {"/host": {}}),
        ):
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.image["Config"][key] = value
                self.container["Config"][key] = value
                self.reject("profile_rejected")

    def test_all_fixed_environment_bindings_are_required_exactly(self):
        original = list(self.image["Config"]["Env"])
        for index, entry in enumerate(original):
            with self.subTest(variable=entry.split("=", 1)[0]):
                self.image, self.container = metadata()
                env = list(original)
                env[index] = entry.split("=", 1)[0] + "=wrong"
                self.image["Config"]["Env"] = self.container["Config"]["Env"] = env
                self.reject("profile_rejected")

    def test_environment_duplicate_keys_malformed_rows_and_budgets_are_rejected(self):
        original = self.image["Config"]["Env"]
        variants = [
            original + [original[0]],
            original + ["NO_EQUALS"],
            original + [False],
            original + ["X=" + "a" * 4095],
            ["X=x"] * 129,
            {},
            None,
        ]
        for env in variants:
            with self.subTest(kind=type(env).__name__, size=len(env) if env is not None else 0):
                self.image, self.container = metadata()
                self.image["Config"]["Env"] = self.container["Config"]["Env"] = env
                self.reject("profile_rejected")

    def test_capability_privilege_network_and_namespace_changes_are_rejected(self):
        changes = {
            "ReadonlyRootfs": False,
            "Privileged": True,
            "AutoRemove": True,
            "NetworkMode": "host",
            "CapDrop": [],
            "SecurityOpt": ["seccomp=unconfined"],
            "IpcMode": "host",
            "PidMode": "host",
            "UTSMode": "host",
            "UsernsMode": "host",
            "CgroupParent": "foreign",
        }
        for key, value in changes.items():
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.container["HostConfig"][key] = value
                self.reject("profile_rejected")

    def test_host_mount_device_capability_link_and_port_injections_are_rejected(self):
        for key in (
            "Binds",
            "Mounts",
            "Devices",
            "DeviceRequests",
            "DeviceCgroupRules",
            "CapAdd",
            "Links",
            "PortBindings",
        ):
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.container["HostConfig"][key] = ["injected"]
                self.reject("profile_rejected")

    def test_resource_caps_are_exact_integer_not_boolean_or_float(self):
        for key in ("Memory", "MemorySwap", "ShmSize", "PidsLimit", "NanoCpus"):
            original = self.container["HostConfig"][key]
            for value in (original + 1, float(original), True, str(original)):
                with self.subTest(field=key, value=value):
                    self.image, self.container = metadata()
                    self.container["HostConfig"][key] = value
                    self.reject("profile_rejected")

    def test_tmpfs_restart_policy_and_ulimits_cannot_change(self):
        for key, value in (
            ("Tmpfs", {}),
            ("RestartPolicy", {"Name": "always"}),
            ("Ulimits", []),
            (
                "Ulimits",
                [
                    {"Name": "core", "Soft": False, "Hard": False},
                    {"Name": "nofile", "Soft": 1024, "Hard": 1024},
                ],
            ),
        ):
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.container["HostConfig"][key] = value
                self.reject("profile_rejected")

    def test_optional_inspected_tmpfs_and_reversed_ulimits_are_supported(self):
        self.container["Mounts"] = [{"Type": "tmpfs", "Destination": "/tmp", "Source": ""}]
        self.container["HostConfig"]["Ulimits"].reverse()
        self.container["HostConfig"]["SecurityOpt"] = ["no-new-privileges:true"]
        self.assertEqual(self.verify(), admission.SoftwareBinding())

    def test_inspected_mounts_cannot_add_sources_destinations_or_extra_entries(self):
        for mounts in (
            {},
            [{"Type": "bind", "Destination": "/tmp", "Source": "/host"}],
            [{"Type": "tmpfs", "Destination": "/other"}],
            [{"Type": "tmpfs", "Destination": "/tmp", "Source": "/host"}],
            [{"Type": "tmpfs", "Destination": "/tmp"}] * 2,
        ):
            with self.subTest(kind=type(mounts).__name__):
                self.container["Mounts"] = mounts
                self.reject("profile_rejected")

    def test_missing_nested_fields_and_wrong_types_are_sanitized(self):
        for key, value in (("State", None), ("Config", []), ("HostConfig", {}), ("Mounts", [None])):
            with self.subTest(field=key):
                self.image, self.container = metadata()
                self.container[key] = value
                self.reject("inspect_failed")

    def test_invalid_duplicate_nonfinite_and_wrong_shape_json_is_rejected(self):
        for raw in (
            b'[{"Id":1,"Id":2}]',
            b'[{"nested":{"x":1,"x":2}}]',
            b'[{"value":NaN}]',
            b'[{"value":Infinity}]',
            b"{}",
            b"[]",
            b"[{},{}]",
            b"[1]",
        ):
            with self.subTest(raw=raw):
                self.inspector.side_effect = None
                self.inspector.return_value = raw
                self.reject("inspect_shape_rejected")

    def test_invalid_json_parser_details_are_not_exposed(self):
        self.inspector.side_effect = None
        self.inspector.return_value = b'{"' + PRIVATE_TEXT.encode("ascii")
        self.reject("inspect_failed")

    def test_inspect_raw_byte_budget_is_enforced(self):
        for raw in (b"", "[]", bytearray(b"[]"), b" " * (admission.MAXIMUM_INSPECT_BYTES + 1)):
            with self.subTest(kind=type(raw).__name__, size=len(raw)):
                self.inspector.side_effect = None
                self.inspector.return_value = raw
                self.reject("inspect_budget_exhausted")

    def test_inspect_exception_details_do_not_escape(self):
        self.inspector.side_effect = RuntimeError(PRIVATE_TEXT)
        self.reject("inspect_failed")

    def test_allowlisted_inspect_error_survives_without_original_context(self):
        self.inspector.side_effect = admission.OciAdmissionError("inspect_budget_exhausted")
        self.reject("inspect_budget_exhausted")

    def test_deadline_is_rechecked_after_both_metadata_reads(self):
        self.clock.monotonic.side_effect = (100.0, 110.0)
        self.reject("deadline_exhausted")
        self.assertEqual(self.inspector.call_count, 2)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.os = types.SimpleNamespace(
            getpid=Mock(return_value=1234), getppid=Mock(return_value=1233)
        )
        main = object()
        self.threads = types.SimpleNamespace(
            current_thread=Mock(return_value=main),
            main_thread=Mock(return_value=main),
            active_count=Mock(return_value=1),
        )
        self.clock = types.SimpleNamespace(monotonic=Mock(return_value=100.0))
        self.verify = Mock(return_value=admission.SoftwareBinding())
        self.continuation = Mock(return_value=None)
        for name, value in (
            ("os", self.os),
            ("threading", self.threads),
            ("time", self.clock),
            ("_runtime", Mock()),
            ("verify_created_oci", self.verify),
        ):
            selected = patch.object(admission, name, value)
            selected.start()
            self.addCleanup(selected.stop)

    def make(self, **overrides):
        arguments = {
            "container_id": CONTAINER_ID,
            "expected_parent_pid": 1233,
            "deadline": 110.0,
            "continuation": self.continuation,
        }
        arguments.update(overrides)
        return admission.VerifiedOciHandoff(**arguments)

    def status(self, selected, state, code, counts):
        result = selected.status()
        self.assertEqual(set(result), {"schema", "state", "code", *COUNTERS, *admission.NONCLAIMS})
        self.assertEqual(result["schema"], admission.SCHEMA)
        self.assertEqual((result["state"], result["code"]), (state, code))
        self.assertEqual(tuple(result[key] for key in COUNTERS), counts)
        for name in COUNTERS:
            self.assertIs(type(result[name]), int)
        for name in admission.NONCLAIMS:
            self.assertIs(result[name], False)
        for text in (CONTAINER_ID, admission.IMAGE_ID, PRIVATE_TEXT, "1234", "1233", "/"):
            self.assertNotIn(text, json.dumps(result))
        return result

    def reject(self, selected, code, counts=(0, 0, 0, 0)):
        with self.assertRaises(admission.OciAdmissionError) as caught:
            selected()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.status(selected, "failed", code, counts)

    def test_constructor_performs_no_inspect_or_continuation(self):
        selected = self.make()
        self.verify.assert_not_called()
        self.continuation.assert_not_called()
        self.status(selected, "ready", "not_started", (0, 0, 0, 0))

    def test_status_is_fresh_and_never_grants_execution_claims(self):
        selected = self.make()
        changed = selected.status()
        for name in admission.NONCLAIMS:
            changed[name] = True
        self.status(selected, "ready", "not_started", (0, 0, 0, 0))

    def test_valid_handoff_verifies_before_continuation_once(self):
        selected = self.make()
        events = []

        def verify(*args, **kwargs):
            events.append("verify")
            self.status(selected, "verifying", "not_started", (1, 0, 0, 0))
            return admission.SoftwareBinding()

        def continue_metadata(binding):
            events.append("continuation")
            self.assertEqual(binding, admission.SoftwareBinding())
            self.status(selected, "handing_off", "not_started", (1, 1, 1, 0))

        self.verify.side_effect = verify
        self.continuation.side_effect = continue_metadata
        self.assertIsNone(selected())
        self.assertEqual(events, ["verify", "continuation"])
        self.verify.assert_called_once_with(CONTAINER_ID, deadline=110.0)
        self.status(selected, "handed_off", "completed", (1, 1, 1, 1))

    def test_continuation_receives_fixed_source_identity_only_once(self):
        selected = self.make()
        identities = []
        self.continuation.side_effect = lambda binding: identities.append(
            binding.source_identity_sha256(),
        )
        selected()
        self.reject(selected, "already_consumed", (1, 1, 1, 1))
        self.assertEqual(identities, [admission.SoftwareBinding().source_identity_sha256()])

    def test_success_is_consumed_and_cannot_be_retried(self):
        selected = self.make()
        selected()
        self.reject(selected, "already_consumed", (1, 1, 1, 1))
        self.assertEqual(self.verify.call_count, 1)
        self.assertEqual(self.continuation.call_count, 1)

    def test_reinitialization_cannot_reset_consumed_instance(self):
        selected = self.make()
        selected()
        with self.assertRaises(admission.OciAdmissionError) as caught:
            selected.__init__(
                container_id=CONTAINER_ID,
                expected_parent_pid=1233,
                deadline=110.0,
                continuation=self.continuation,
            )
        self.assertEqual(caught.exception.code, "configuration_rejected")
        self.reject(selected, "already_consumed", (1, 1, 1, 1))
        self.assertEqual(self.verify.call_count, 1)

    def test_wrong_parent_before_verification_never_calls_it(self):
        selected = self.make()
        self.os.getppid.return_value = 2222
        self.reject(selected, "parent_lost")
        self.verify.assert_not_called()
        self.continuation.assert_not_called()

    def test_parent_loss_during_verification_prevents_handoff(self):
        selected = self.make()
        self.verify.side_effect = lambda *args, **kwargs: (
            setattr(self.os.getppid, "return_value", 2222) or admission.SoftwareBinding()
        )
        self.reject(selected, "parent_lost", (1, 0, 0, 0))
        self.continuation.assert_not_called()

    def test_expired_before_verification_never_calls_it(self):
        selected = self.make()
        self.clock.monotonic.return_value = 110.0
        self.reject(selected, "deadline_exhausted")
        self.verify.assert_not_called()

    def test_deadline_expiring_during_verification_prevents_handoff(self):
        selected = self.make()
        self.verify.side_effect = lambda *args, **kwargs: (
            setattr(self.clock.monotonic, "return_value", 110.0) or admission.SoftwareBinding()
        )
        self.reject(selected, "deadline_exhausted", (1, 0, 0, 0))
        self.continuation.assert_not_called()

    def test_owner_pid_and_thread_are_checked_before_verification(self):
        for target, value in (
            (self.os.getpid, 4321),
            (self.threads.current_thread, object()),
            (self.threads.active_count, 2),
        ):
            original = target.return_value
            selected = self.make()
            target.return_value = value
            self.reject(selected, "owner_rejected")
            target.return_value = original
        self.verify.assert_not_called()

    def test_runtime_rejection_is_consumed(self):
        selected = self.make()
        admission._runtime.side_effect = admission.OciAdmissionError("runtime_rejected")
        self.reject(selected, "runtime_rejected")
        admission._runtime.side_effect = None
        self.reject(selected, "already_consumed")
        self.verify.assert_not_called()

    def test_verification_error_is_fixed_and_continuation_never_runs(self):
        selected = self.make()
        self.verify.side_effect = RuntimeError(PRIVATE_TEXT)
        self.reject(selected, "inspect_failed", (1, 0, 0, 0))
        self.continuation.assert_not_called()

    def test_allowlisted_verification_failure_is_preserved(self):
        selected = self.make()
        self.verify.side_effect = admission.OciAdmissionError("container_rejected")
        self.reject(selected, "container_rejected", (1, 0, 0, 0))

    def test_wrong_type_or_substituted_binding_never_reaches_continuation(self):
        for result in (None, {}, True, admission.SoftwareBinding(source_commit="b" * 40)):
            with self.subTest(kind=type(result).__name__):
                selected = self.make()
                self.verify.return_value = result
                self.reject(selected, "image_rejected", (1, 0, 0, 0))
        self.continuation.assert_not_called()

    def test_continuation_failure_is_fixed_nonretryable_and_not_completed(self):
        selected = self.make()
        self.continuation.side_effect = RuntimeError(PRIVATE_TEXT)
        self.reject(selected, "handoff_failed", (1, 1, 1, 0))
        self.reject(selected, "already_consumed", (1, 1, 1, 0))
        self.continuation.assert_called_once()

    def test_continuation_must_return_none_not_a_permission_value(self):
        for result in (False, True, 0, {}, admission.SoftwareBinding()):
            with self.subTest(kind=type(result).__name__):
                selected = self.make()
                self.continuation.return_value = result
                self.reject(selected, "handoff_return_rejected", (1, 1, 1, 0))

    def test_parent_loss_after_continuation_does_not_count_completion(self):
        selected = self.make()
        self.continuation.side_effect = lambda binding: setattr(
            self.os.getppid, "return_value", 2222
        )
        self.reject(selected, "parent_lost", (1, 1, 1, 0))

    def test_deadline_after_continuation_does_not_count_completion(self):
        selected = self.make()
        self.continuation.side_effect = lambda binding: setattr(
            self.clock.monotonic,
            "return_value",
            110.0,
        )
        self.reject(selected, "deadline_exhausted", (1, 1, 1, 0))

    def test_caught_reentry_during_verification_poison_outer_handoff(self):
        selected = self.make()

        def reenter(*args, **kwargs):
            with self.assertRaises(admission.OciAdmissionError):
                selected()
            return admission.SoftwareBinding()

        self.verify.side_effect = reenter
        self.reject(selected, "already_consumed", (1, 0, 0, 0))
        self.continuation.assert_not_called()

    def test_caught_reentry_during_continuation_cannot_be_turned_into_success(self):
        selected = self.make()

        def reenter(binding):
            with self.assertRaises(admission.OciAdmissionError):
                selected()

        self.continuation.side_effect = reenter
        self.reject(selected, "already_consumed", (1, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
