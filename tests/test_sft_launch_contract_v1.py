"""Synthetic, metadata-only launch-contract tests; never open model or data files."""

from __future__ import annotations

import copy
import hashlib
import unittest

from lumi import posttraining_sft_launch_contract_v1 as launch


def fake_sha(label: str) -> str:
    return hashlib.sha256(("SYNTHETIC-ONLY:" + label).encode("ascii")).hexdigest()


def synthetic_inputs() -> tuple[dict, dict, dict]:
    base = {
        "model_id": "lumi-4b-base",
        "checkpoint_stage": "base",
        "source_commit": "a" * 40,
        "checkpoint": {
            "step": 76294,
            "training_tokens": 80_000_000_000,
            "manifest_sha256": fake_sha("dcp"),
        },
        "tokenizer": {"manifest_sha256": fake_sha("tokenizer")},
        "s3_backup": {
            "status": "verified",
            "bucket": "synthetic-no-access",
            "prefix": "synthetic/base-only",
        },
    }
    cpu = {
        "encoded_directory": "/synthetic/origin/encoded",
        "source_checkpoint": "/synthetic/tiny/not-the-real-base",
        "source_checkpoint_manifest_sha256": fake_sha("cpu-tiny"),
        **{field: fake_sha(field) for field in launch.FIVE_HASHES},
    }
    profile = {
        "world_size": 2,
        "strategy": "fsdp_full_shard",
        "device": "cuda",
        "backend": "nccl",
        "precision": "bf16",
        "attention_implementation": "eager",
        "record_limit": 12,
        "per_rank_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "context_window": 2048,
        "target_steps": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "maximum_gradient_norm": 1.0,
        "lr_schedule": "cosine",
        "warmup_steps": 0,
        "minimum_learning_rate_ratio": 0.1,
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "checkpoint_steps": 1,
        "seed": 2027,
        "pad_to_multiple_of": 8,
        "gradient_sync_mode": "last_microbatch",
        "sequence": [
            {"run": "reference", "from_step": 0, "stop_step": 2, "resume": False},
            {"run": "resumed", "from_step": 0, "stop_step": 1, "resume": False},
            {"run": "resumed", "from_step": 1, "stop_step": 2, "resume": True},
        ],
    }
    return base, cpu, profile


def synthetic_reference() -> dict:
    return launch.make_draft(*synthetic_inputs())


def fill_synthetic_slots(reference: dict) -> dict:
    contract = copy.deepcopy(reference)
    target = {
        "host_id": "SYNTHETIC-HOST",
        "gpu_uuids": ["GPU-SYNTHETIC-0", "GPU-SYNTHETIC-1"],
        "runtime_lock_sha256": fake_sha("runtime"),
        "sft_source_commit": "b" * 40,
        "python_path": "/synthetic/runtime/python",
        "checkpoint_directory": "/synthetic/import/checkpoint",
        "encoded_directory": "/synthetic/relocation/encoded",
        "private_output_root": "/synthetic/output/private",
        "authorization_path": "/synthetic/approval/authorization.json",
        "local_rendezvous": {
            "address": "127.0.0.1",
            "port": 29501,
            "node_count": 1,
            "maximum_restarts": 0,
        },
    }
    contract["target"] = target
    base = contract["source"]["historical_binding"]
    contract["import_receipt"] = {
        "file_sha256": fake_sha("import-file"),
        "internal_sha256": fake_sha("import-internal"),
        "verification_file_sha256": fake_sha("import-verification"),
        "host_id": target["host_id"],
        "checkpoint_directory": target["checkpoint_directory"],
        "real_4b_import": True,
        "original_dcp_manifest_sha256": base["checkpoint"]["manifest_sha256"],
        "hf_checksum_manifest_sha256": fake_sha("hf-manifest"),
        "tokenizer_manifest_sha256": base["tokenizer"]["manifest_sha256"],
        "final_base_binding_sha256": launch.BASE_SHA,
    }
    origin = contract["data"]["origin_loader_hashes"]
    contract["relocation_receipt"] = {
        "file_sha256": fake_sha("relocation-file"),
        "internal_sha256": fake_sha("relocation-internal"),
        "verification_file_sha256": fake_sha("relocation-verification"),
        "host_id": target["host_id"],
        "encoded_directory": target["encoded_directory"],
        "content_verified": True,
        "private_modes_verified": True,
        "origin_metadata_inventory_sha256": origin["metadata_inventory_sha256"],
        "loader_hashes": {**origin, "metadata_inventory_sha256": fake_sha("new-host-metadata")},
    }
    contract["cuda_acceptance"] = {
        "criteria_file_sha256": fake_sha("criteria"),
        "host_id": target["host_id"],
        "runtime_lock_sha256": target["runtime_lock_sha256"],
        "criteria": [
            "masking",
            "precision",
            "memory",
            "import",
            "fresh_process_resume",
            "fault_handling",
        ],
    }
    counts = launch.geometry(contract["profile"])
    contract["resource_caps"] = {
        **{field: 10_000_000 for field in launch.CAP_FIELDS},
        "maximum_combined_runtime_seconds": 60,
        "total_host_hourly_rate_microusd": 3_600_000,
        "maximum_cost_microusd": 60_000,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4,
        "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return contract


def synthetic_approval(contract: dict) -> dict:
    return {
        "schema_version": launch.AUTH_SCHEMA,
        "status": "owner_approved",
        "run_kind": "sft_gpu_smoke_and_fresh_resume_only",
        "scope": launch.SCOPE,
        "subject_sha256": launch.authorization_subject(contract),
        "owner_approval_text_sha256": fake_sha("synthetic-not-real-approval"),
        "not_before_unix": 1000,
        "expires_unix": 2000,
        "gpu_execution_approved": True,
        "production_eligible": False,
    }


class LaunchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = synthetic_reference()

    def filled(self) -> dict:
        return fill_synthetic_slots(self.reference)

    def assert_rejected(self, contract: dict, code: str | None = None) -> None:
        with self.assertRaises(ValueError) as caught:
            launch.analyze(contract, self.reference)
        if code is not None:
            self.assertEqual(str(caught.exception), code)

    def test_draft_keeps_s3_base_not_cpu_tiny_identity(self):
        base, cpu, profile = synthetic_inputs()
        contract = launch.make_draft(base, cpu, profile)
        self.assertEqual(contract["source"]["historical_binding"], base)
        self.assertNotIn(cpu["source_checkpoint"], launch.canonical(contract).decode())
        self.assertNotIn(
            cpu["source_checkpoint_manifest_sha256"], launch.canonical(contract).decode()
        )
        self.assertIs(contract["source"]["remote_availability_rechecked"], False)

    def test_rejects_nonfinal_base_identity(self):
        for field, value in (("model_id", "tiny-cpu"), ("checkpoint_stage", "sft")):
            with self.subTest(field=field):
                base, cpu, profile = synthetic_inputs()
                base[field] = value
                with self.assertRaisesRegex(ValueError, "final_base_identity_changed"):
                    launch.make_draft(base, cpu, profile)

    def test_rejects_changed_pretraining_boundary(self):
        for field in ("step", "training_tokens"):
            with self.subTest(field=field):
                base, cpu, profile = synthetic_inputs()
                base["checkpoint"][field] += 1
                with self.assertRaisesRegex(ValueError, "final_base_boundary_changed"):
                    launch.make_draft(base, cpu, profile)

    def test_rejects_unverified_historical_backup(self):
        base, cpu, profile = synthetic_inputs()
        base["s3_backup"]["status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "historical_backup_not_verified"):
            launch.make_draft(base, cpu, profile)

    def test_carries_exactly_five_origin_hashes(self):
        origin = self.reference["data"]["origin_loader_hashes"]
        self.assertEqual(set(origin), set(launch.FIVE_HASHES))
        self.assertIs(self.reference["data"]["origin_metadata_binding_is_portable"], False)

    def test_rejects_invalid_origin_digest(self):
        for value in (True, "A" * 64, "0" * 63, "g" * 64, None):
            with self.subTest(value=value):
                base, cpu, profile = synthetic_inputs()
                cpu["metadata_inventory_sha256"] = value
                with self.assertRaisesRegex(ValueError, "invalid_sha256"):
                    launch.make_draft(base, cpu, profile)

    def test_draft_deep_copies_inputs(self):
        base, cpu, profile = synthetic_inputs()
        result = launch.make_draft(base, cpu, profile)
        base["s3_backup"]["prefix"] = "mutated"
        profile["sequence"][0]["stop_step"] = 99
        self.assertEqual(
            result["source"]["historical_binding"]["s3_backup"]["prefix"], "synthetic/base-only"
        )
        self.assertEqual(result["profile"]["sequence"][0]["stop_step"], 2)

    def test_empty_draft_reports_all_unbound_gates(self):
        report = launch.analyze(self.reference, self.reference)
        self.assertEqual(
            report["pending_gates"],
            [slot + "_unbound" for slot in launch.SLOTS] + list(launch.ALWAYS_BLOCKED),
        )
        self.assertIsNone(report["worker_option_previews"])

    def test_frozen_top_level_scope_and_execution_fields(self):
        for field, value in (
            ("schema_version", "other"),
            ("scope", "production"),
            ("operation", "full_sft"),
            ("execution_enabled", True),
            ("production_eligible", True),
            ("execution_adapter", "worker"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.reference)
                changed[field] = value
                self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_contract_schema_rejects_extra_and_missing_fields(self):
        extra = {**self.reference, "surprise": True}
        self.assert_rejected(extra, "contract_schema")
        missing = copy.deepcopy(self.reference)
        del missing["target"]
        self.assert_rejected(missing, "contract_schema")

    def test_source_binding_is_frozen(self):
        changed = copy.deepcopy(self.reference)
        changed["source"]["historical_binding"]["s3_backup"]["prefix"] = "other"
        self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_data_binding_is_frozen(self):
        for field in launch.FIVE_HASHES:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.reference)
                changed["data"]["origin_loader_hashes"][field] = fake_sha("changed")
                self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_training_profile_is_frozen(self):
        for field, value in (
            ("learning_rate", 0.2),
            ("record_limit", 13),
            ("target_steps", 3),
            ("strategy", "ddp"),
            ("precision", "fp32"),
            ("seed", 12),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.reference)
                changed["profile"][field] = value
                self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_prohibitions_cannot_be_relaxed(self):
        for field in launch.PROHIBITIONS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.reference)
                changed["prohibitions"][field] = False
                self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_frozen_profile_preserves_numeric_json_types(self):
        changed = copy.deepcopy(self.reference)
        changed["profile"]["maximum_gradient_norm"] = 1
        self.assert_rejected(changed, "frozen_contract_field_changed")

    def test_geometry_counts_reference_plus_fresh_resume(self):
        counts = launch.geometry(self.reference["profile"])
        expected = {
            "sequential_invocations": 3,
            "global_optimizer_steps": 4,
            "rank_optimizer_steps": 8,
            "logical_sample_events": 8,
            "maximum_unique_train_records": 12,
            "rank_local_input_loads": 6,
            "maximum_encoded_index_rows": 72,
            "maximum_selected_binary_bytes": 72 * 2048 * 8,
            "maximum_selected_index_bytes": 72 * 176,
        }
        for field, value in expected.items():
            self.assertEqual(counts[field], value)
        self.assertFalse(counts["is_total_io_or_cost_budget"])
        self.assertFalse(counts["includes_checkpoint_hashing_or_final_digest"])

    def test_sequence_cannot_retry_reorder_or_extend(self):
        original = self.reference["profile"]["sequence"]
        variants = [original[:-1], list(reversed(original)), original + [original[-1]]]
        for sequence in variants:
            with self.subTest(sequence=sequence):
                profile = copy.deepcopy(self.reference["profile"])
                profile["sequence"] = sequence
                with self.assertRaisesRegex(ValueError, "smoke_resume_sequence_changed"):
                    launch.geometry(profile)

    def test_geometry_rejects_unsafe_rank_and_record_bounds(self):
        for field, value in (
            ("world_size", 1),
            ("world_size", True),
            ("record_limit", 13),
            ("record_limit", 0),
            ("per_rank_batch_size", 7),
            ("gradient_accumulation_steps", -1),
            ("context_window", 2**63),
        ):
            with self.subTest(field=field, value=value):
                profile = copy.deepcopy(self.reference["profile"])
                profile[field] = value
                with self.assertRaises(ValueError):
                    launch.geometry(profile)

    def test_previews_need_import_and_relocation_slots(self):
        for slot in ("target", "import_receipt", "relocation_receipt", "resource_caps"):
            with self.subTest(slot=slot):
                contract = self.filled()
                contract[slot] = None
                self.assertIsNone(launch.argument_previews(contract))

    def test_valid_synthetic_target_is_metadata_only(self):
        contract = self.filled()
        launch.validate_target(contract["target"], contract["profile"])
        self.assertFalse(launch.analyze(contract, self.reference)["receipt_contents_verified"])

    def test_target_exact_schema(self):
        for mode in ("extra", "missing"):
            with self.subTest(mode=mode):
                contract = self.filled()
                if mode == "extra":
                    contract["target"]["shell"] = "bash"
                else:
                    del contract["target"]["host_id"]
                self.assert_rejected(contract, "target_schema")

    def test_target_rejects_relative_broad_normalized_and_control_paths(self):
        for path in (
            "relative/path",
            "/",
            "/tmp",
            "/a/../b",
            "/a//b",
            "//a/b",
            "/a/b/",
            "/a/b\n",
            "/a/\x00b",
            "/a/\x7fb",
        ):
            with self.subTest(path=path):
                contract = self.filled()
                contract["target"]["python_path"] = path
                self.assert_rejected(contract)

    def test_target_rejects_pairwise_overlap(self):
        for path in (
            "/synthetic/import/checkpoint",
            "/synthetic/import",
            "/synthetic/import/checkpoint/output",
        ):
            with self.subTest(path=path):
                contract = self.filled()
                contract["target"]["private_output_root"] = path
                self.assert_rejected(contract, "target_paths_overlap")

    def test_target_rejects_invalid_host_and_gpu_labels(self):
        for label in ("", "GPU with spaces", "GPU\n0", "$(command)", True):
            with self.subTest(label=label):
                contract = self.filled()
                contract["target"]["gpu_uuids"][0] = label
                self.assert_rejected(contract, "invalid_identity")

    def test_target_rejects_duplicate_or_wrong_count_gpus(self):
        for identities in (["GPU-0"], ["GPU-0", "GPU-0"], ["GPU-0", "GPU-1", "GPU-2"]):
            with self.subTest(identities=identities):
                contract = self.filled()
                contract["target"]["gpu_uuids"] = identities
                self.assert_rejected(contract)

    def test_target_runtime_hash_and_sft_commit_are_strict(self):
        for field, value in (
            ("runtime_lock_sha256", "x" * 64),
            ("sft_source_commit", "b" * 39),
            ("sft_source_commit", "B" * 40),
            ("sft_source_commit", True),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["target"][field] = value
                self.assert_rejected(contract)

    def test_local_rendezvous_rejects_external_multinode_or_retries(self):
        for field, value in (
            ("address", "0.0.0.0"),
            ("address", "remote.example"),
            ("port", True),
            ("port", 1023),
            ("port", 65536),
            ("node_count", 2),
            ("node_count", True),
            ("maximum_restarts", 1),
            ("maximum_restarts", False),
        ):
            with self.subTest(field=field, value=value):
                contract = self.filled()
                contract["target"]["local_rendezvous"][field] = value
                self.assert_rejected(contract, "nonlocal_or_retrying_rendezvous")

    def test_sequence_rejects_integer_resume_flag(self):
        profile = copy.deepcopy(self.reference["profile"])
        profile["sequence"][0]["resume"] = 0
        with self.assertRaisesRegex(ValueError, "smoke_resume_sequence_changed"):
            launch.geometry(profile)

    def test_preview_timeout_is_explicit_and_bounded(self):
        contract = self.filled()
        for argv in launch.argument_previews(contract):
            self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "60")
        contract["resource_caps"]["maximum_combined_runtime_seconds"] = 2400
        for argv in launch.argument_previews(contract):
            self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "1800")

    def test_receipt_requires_target(self):
        for slot in ("import_receipt", "relocation_receipt", "cuda_acceptance"):
            with self.subTest(slot=slot):
                contract = copy.deepcopy(self.reference)
                contract[slot] = self.filled()[slot]
                self.assert_rejected(contract, "receipt_requires_target")

    def test_import_requires_real_claim_matching_target(self):
        for field, value in (
            ("real_4b_import", False),
            ("real_4b_import", 1),
            ("host_id", "OTHER"),
            ("checkpoint_directory", "/other/checkpoint"),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["import_receipt"][field] = value
                self.assert_rejected(contract)

    def test_import_rejects_lineage_substitution(self):
        for field in (
            "original_dcp_manifest_sha256",
            "tokenizer_manifest_sha256",
            "final_base_binding_sha256",
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["import_receipt"][field] = fake_sha("wrong-lineage")
                self.assert_rejected(contract, "import_lineage_mismatch")

    def test_original_dcp_manifest_cannot_be_hf_manifest(self):
        contract = self.filled()
        receipt = contract["import_receipt"]
        receipt["hf_checksum_manifest_sha256"] = receipt["original_dcp_manifest_sha256"]
        self.assert_rejected(contract, "dcp_manifest_is_not_hf_manifest")

    def test_relocation_requires_all_five_hashes_and_no_extra(self):
        for field in launch.FIVE_HASHES:
            with self.subTest(field=field):
                contract = self.filled()
                del contract["relocation_receipt"]["loader_hashes"][field]
                self.assert_rejected(contract, "five_loader_hashes_required")
        contract = self.filled()
        contract["relocation_receipt"]["loader_hashes"]["extra"] = fake_sha("extra")
        self.assert_rejected(contract, "five_loader_hashes_required")

    def test_relocation_preserves_four_content_hashes(self):
        for field in set(launch.FIVE_HASHES) - {"metadata_inventory_sha256"}:
            with self.subTest(field=field):
                contract = self.filled()
                contract["relocation_receipt"]["loader_hashes"][field] = fake_sha("changed-content")
                self.assert_rejected(contract, "relocated_content_binding_changed")

    def test_relocation_requires_new_metadata_bound_to_origin(self):
        contract = self.filled()
        receipt = contract["relocation_receipt"]
        receipt["loader_hashes"]["metadata_inventory_sha256"] = receipt[
            "origin_metadata_inventory_sha256"
        ]
        self.assert_rejected(contract, "origin_metadata_not_portable")
        contract = self.filled()
        contract["relocation_receipt"]["origin_metadata_inventory_sha256"] = fake_sha(
            "wrong-origin"
        )
        self.assert_rejected(contract, "relocation_origin_metadata_mismatch")

    def test_relocation_requires_private_verified_matching_target(self):
        for field, value in (
            ("content_verified", False),
            ("private_modes_verified", 1),
            ("host_id", "OTHER"),
            ("encoded_directory", "/other/encoded"),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["relocation_receipt"][field] = value
                self.assert_rejected(contract)

    def test_cuda_criteria_must_be_complete_and_target_bound(self):
        for field, value in (
            ("criteria", ["masking"]),
            ("host_id", "OTHER"),
            ("runtime_lock_sha256", fake_sha("wrong-runtime")),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["cuda_acceptance"][field] = value
                self.assert_rejected(contract)

    def test_caps_require_exact_schema(self):
        for mode in ("extra", "missing"):
            with self.subTest(mode=mode):
                contract = self.filled()
                if mode == "extra":
                    contract["resource_caps"]["unbounded"] = True
                else:
                    del contract["resource_caps"]["maximum_log_bytes"]
                self.assert_rejected(contract, "resource_caps_schema")

    def test_caps_reject_bool_nonfinite_negative_fractional_and_overflow(self):
        for field in launch.CAP_FIELDS:
            for value in (True, False, float("inf"), float("nan"), -1, 0, 1.5, 2**63):
                with self.subTest(field=field, value=value):
                    contract = self.filled()
                    contract["resource_caps"][field] = value
                    self.assert_rejected(contract)

    def test_operation_caps_forbid_extra_invocations_network_and_inference(self):
        for field, value in (
            ("maximum_invocations", 4),
            ("maximum_global_optimizer_steps", 5),
            ("maximum_external_network_bytes", 1),
            ("maximum_inference_calls", True),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                contract["resource_caps"][field] = value
                self.assert_rejected(contract, "operation_cap_changed")

    def test_cost_cap_uses_integer_ceiling_not_round_down(self):
        contract = self.filled()
        caps = contract["resource_caps"]
        caps["maximum_combined_runtime_seconds"] = 1
        caps["total_host_hourly_rate_microusd"] = 3601
        caps["maximum_cost_microusd"] = 1
        self.assert_rejected(contract, "runtime_exceeds_cost_cap")
        caps["maximum_cost_microusd"] = 2
        launch.analyze(contract, self.reference)

    def test_rank_multiplied_input_caps_cannot_be_one_rank_only(self):
        for field in ("maximum_encoded_binary_read_bytes", "maximum_encoded_index_read_bytes"):
            with self.subTest(field=field):
                contract = self.filled()
                contract["resource_caps"][field] //= contract["profile"]["world_size"]
                self.assert_rejected(contract, "rank_multiplied_data_read_budget_too_small")

    def test_logs_cannot_exceed_total_private_write_cap(self):
        contract = self.filled()
        contract["resource_caps"]["maximum_log_bytes"] += 1
        self.assert_rejected(contract, "log_cap_exceeds_private_write_cap")

    def test_authorization_rejects_generic_pretraining_run_kinds(self):
        for kind in (
            "lumi_distributed_smoke_v0.1",
            "lumi_4b_learning_pilot_v0.1",
            "lumi_4b_production_v0.1",
        ):
            with self.subTest(kind=kind):
                contract = self.filled()
                auth = synthetic_approval(contract)
                auth["run_kind"] = kind
                contract["execution_authorization"] = auth
                self.assert_rejected(contract, "not_sft_specific_authorization")

    def test_authorization_requires_fully_bound_slots(self):
        for slot in launch.SLOTS[:-1]:
            with self.subTest(slot=slot):
                contract = self.filled()
                contract[slot] = None
                if slot == "target":
                    for receipt in ("import_receipt", "relocation_receipt", "cuda_acceptance"):
                        contract[receipt] = None
                contract["execution_authorization"] = synthetic_approval(contract)
                self.assert_rejected(contract, "approval_requires_fully_bound_inputs")

    def test_authorization_binds_full_subject_not_only_training_profile(self):
        for mutation in ("host", "cap", "receipt"):
            with self.subTest(mutation=mutation):
                contract = self.filled()
                auth = synthetic_approval(contract)
                if mutation == "host":
                    contract["target"]["python_path"] = "/synthetic/changed/python"
                elif mutation == "cap":
                    contract["resource_caps"]["maximum_cost_microusd"] += 1
                else:
                    contract["import_receipt"]["file_sha256"] = fake_sha("new-receipt")
                contract["execution_authorization"] = auth
                self.assert_rejected(contract, "authorization_subject_mismatch")

    def test_authorization_time_window_is_half_open_and_strict(self):
        contract = self.filled()
        auth = synthetic_approval(contract)
        expected = launch.digest(auth)
        self.assertTrue(launch.validate_authorization(auth, contract, expected, 1000))
        self.assertTrue(launch.validate_authorization(auth, contract, expected, 1999))
        for now in (999, 2000, True, 0):
            with self.subTest(now=now):
                with self.assertRaises(ValueError):
                    launch.validate_authorization(auth, contract, expected, now)
        auth["expires_unix"] = auth["not_before_unix"]
        with self.assertRaisesRegex(ValueError, "authorization_time_window"):
            launch.validate_authorization(auth, contract, None, None)

    def test_authorization_needs_external_digest_and_current_time(self):
        contract = self.filled()
        auth = synthetic_approval(contract)
        self.assertFalse(launch.validate_authorization(auth, contract, None, 1500))
        self.assertFalse(launch.validate_authorization(auth, contract, launch.digest(auth), None))
        with self.assertRaisesRegex(ValueError, "authorization_hash_mismatch"):
            launch.validate_authorization(auth, contract, fake_sha("wrong-approval"), 1500)
        contract["execution_authorization"] = auth
        self.assertIn(
            "execution_approval_not_externally_bound",
            launch.analyze(contract, self.reference)["pending_gates"],
        )

    def test_strict_json_rejects_ambiguous_or_nonfinite_input(self):
        for raw in (
            b'{"x":1,"x":2}',
            b'{"x":{"a":1,"a":2}}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b"[]",
            b"null",
            b"{broken",
            b'{"x":"\xff"}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    launch.strict_json(raw)
        self.assertEqual(launch.strict_json(b'{"x":1}'), {"x": 1})

    def test_populated_synthetic_claims_never_enable_execution(self):
        contract = self.filled()
        contract["execution_authorization"] = synthetic_approval(contract)
        report = launch.analyze(
            contract,
            self.reference,
            expected_authorization_sha256=launch.digest(contract["execution_authorization"]),
            now_unix=1500,
        )
        self.assertEqual(report["pending_gates"], list(launch.ALWAYS_BLOCKED))
        for flag in (
            "execution_permitted",
            "gpu_launch_ready",
            "production_eligible",
            "resource_caps_enforced_by_this_compiler",
            "receipt_contents_verified",
        ):
            self.assertIs(report[flag], False)
        self.assertIsNone(report["executable_argv"])
        self.assertIs(report["previews_are_not_dispatchable"], True)

    def test_previews_carry_relocated_hashes_and_separate_sft_commit(self):
        contract = self.filled()
        previews = launch.analyze(contract, self.reference)["worker_option_previews"]
        self.assertEqual(len(previews), 3)
        for index, argv in enumerate(previews):
            self.assertTrue(argv[0].startswith("--"))
            self.assertNotIn("--keep-checkpoints", argv)
            self.assertIn("--skip-final-model-digest", argv)
            self.assertEqual("--resume" in argv, index == 2)
            self.assertEqual(argv[argv.index("--source-commit") + 1], "b" * 40)
            self.assertEqual(
                argv[argv.index("--source-checkpoint-manifest-sha256") + 1],
                contract["import_receipt"]["hf_checksum_manifest_sha256"],
            )
            for flag, field in (
                ("inventory", "inventory_sha256"),
                ("manifest-file", "manifest_file_sha256"),
                ("manifest", "manifest_internal_sha256"),
                ("metadata-inventory", "metadata_inventory_sha256"),
                ("seal", "seal_internal_sha256"),
            ):
                option = "--encoded-shards-" + flag + "-sha256"
                self.assertEqual(
                    argv[argv.index(option) + 1],
                    contract["relocation_receipt"]["loader_hashes"][field],
                )

    def test_deny_execution_is_unconditional(self):
        with self.assertRaisesRegex(PermissionError, "execution_disabled_in_local_development_v1"):
            launch.deny_execution()

    def test_authorization_schema_scope_and_boolean_flags_are_strict(self):
        for field, value in (
            ("schema_version", "pretraining"),
            ("status", "draft"),
            ("scope", "production"),
            ("gpu_execution_approved", 1),
            ("production_eligible", 0),
            ("owner_approval_text_sha256", "invalid"),
        ):
            with self.subTest(field=field):
                contract = self.filled()
                auth = synthetic_approval(contract)
                auth[field] = value
                contract["execution_authorization"] = auth
                self.assert_rejected(contract)
        contract = self.filled()
        contract["execution_authorization"] = {**synthetic_approval(contract), "extra": True}
        self.assert_rejected(contract, "authorization_schema")


if __name__ == "__main__":
    unittest.main()
