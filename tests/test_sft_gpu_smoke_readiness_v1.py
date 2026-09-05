"""Synthetic-only checks: no real data, checkpoint, torch, network or GPU access."""

import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_gpu_smoke_readiness_v1 as m


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def file(self, name="metadata.json", content=b"{}"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)
        return path

    def test_cache_counts_physical_reads_once(self):
        self.file()
        reader = m.MetadataReader(self.root, {"metadata.json"})
        reader.read("metadata.json")
        reader.read("metadata.json", hashlib.sha256(b"{}").hexdigest())
        self.assertEqual(reader.bytes_read, 2)

    def test_rejects_nonallowlisted_path_before_open(self):
        reader = m.MetadataReader(self.root, set())
        with patch.object(m.os, "open", side_effect=AssertionError("must not open")):
            with self.assertRaisesRegex(ValueError, "not_allowlisted"):
                reader.read("train-00000.index.jsonl")

    def test_wrong_hash_rejected(self):
        self.file()
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            m.MetadataReader(self.root, {"metadata.json"}).read("metadata.json", "0" * 64)

    def test_oversized_metadata_rejected_before_open(self):
        self.file(content=b"x" * 1_048_577)
        with patch.object(m.os, "open", side_effect=AssertionError("must not open")):
            with self.assertRaisesRegex(ValueError, "file_cap"):
                m.MetadataReader(self.root, {"metadata.json"}).read("metadata.json")

    def test_combined_metadata_cap(self):
        self.file()
        reader = m.MetadataReader(self.root, {"metadata.json"})
        reader.bytes_read = 4_194_304
        with self.assertRaisesRegex(ValueError, "read_cap"):
            reader.read("metadata.json")

    def test_symlink_rejected(self):
        self.file()
        (self.root / "link").symlink_to(self.root / "metadata.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            m.MetadataReader(self.root, {"link"}).read("link")

    def test_hardlink_rejected(self):
        self.file()
        os.link(self.root / "metadata.json", self.root / "link")
        with self.assertRaisesRegex(ValueError, "unsafe_metadata"):
            m.MetadataReader(self.root, {"link"}).read("link")

    def test_parent_symlink_rejected(self):
        (self.root / "real").mkdir()
        (self.root / "link").symlink_to(self.root / "real", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            m.safe_path(self.root, "link/new.json")

    def test_escape_paths_rejected(self):
        for relative in ("../x", "/tmp/x", "a/../../x", ""):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                m.safe_path(self.root, relative)

    def test_internal_hash_tampering(self):
        value = {"count": 1}
        value["report_sha256"] = m.digest(value)
        m.internal_hash(value, "report_sha256")
        value["count"] = 2
        with self.assertRaisesRegex(ValueError, "internal_hash_mismatch"):
            m.internal_hash(value, "report_sha256")

    def test_exact_multi_launch_geometry(self):
        counts = m.geometry(m.DRAFT)
        self.assertEqual(counts["sequential_launches"], 3)
        self.assertEqual(counts["global_optimizer_steps_across_all_paths"], 4)
        self.assertEqual(counts["logical_sample_events_across_all_paths"], 32)
        self.assertEqual(counts["maximum_unique_train_records"], 8)
        self.assertEqual(counts["rank_local_dataset_loads"], 24)
        self.assertEqual(counts["maximum_encoded_index_row_reads"], 192)
        self.assertEqual(counts["maximum_selected_binary_bytes"], 6_291_456)
        self.assertEqual(counts["maximum_selected_index_bytes_including_oversize_probe"], 33_792)

    def test_too_small_or_too_large_prefix_rejected(self):
        for count in (7, 13):
            draft = {**m.DRAFT, "record_limit": count}
            with self.assertRaisesRegex(ValueError, "bounded_prefix"):
                m.geometry(draft)

    def test_boolean_is_not_integer_geometry(self):
        with self.assertRaisesRegex(ValueError, "geometry_integer"):
            m.geometry({**m.DRAFT, "world_size": True})

    def test_launch_draft_is_nonexecuting_and_all_unknown_caps_explicit(self):
        self.assertFalse(any(m.BOUNDARY.values()))
        self.assertIsNone(m.DRAFT["executable_argv"])
        self.assertIsNone(m.DRAFT["launch_authorization"])
        self.assertTrue(all(v is None for k, v in m.DRAFT.items()
                            if k.startswith("maximum_") and k != "maximum_gradient_norm"))
        self.assertEqual(len(m.PENDING), 6)

    def test_static_contract_detection_without_importing_trainer(self):
        worker = b'''from .distributed_training import _validate_execution_scope
_validate_execution_scope()
Model.from_pretrained(path, local_files_only=True)
'''
        runtime = b'def _validate_paid_authorization(): return "generic"'
        exporter = (b'_sha256(checkpoint / "manifest.sha256")\n'
                    b'_write_json(temporary / EXPORT_REPORT_FILENAME, report)')
        self.assertTrue(all(m.code_findings(worker, runtime, exporter).values()))
        with self.assertRaisesRegex(ValueError, "export_manifest_contract_changed"):
            m.code_findings(worker, runtime,
                            b'_write_json(temporary / "manifest.sha256", report)')

    def test_static_contract_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "call_graph_changed"):
            m.code_findings(b"x = 1", b"x = 1", b"x = 1")

    def test_shard_inventory_never_opens_content(self):
        self.root.chmod(0o700)
        self.file("manifest.json")
        self.file("train-00000.tokens.u32le", b"private fixture")
        manifest = {"shards": [{"files": [
            {"path": "train-00000.tokens.u32le", "bytes": 15},
        ]}]}
        # Patch only digest so this test focuses on the stat-only access contract.
        with patch.object(m, "digest", return_value="expected"):
            with patch.object(m.os, "open", side_effect=AssertionError("no content opens")):
                result = m.shard_stat_inventory(self.root, manifest, "expected")
        self.assertEqual(result["files_stat_checked"], 2)
        self.assertFalse(result["content_rehashed"])

    def test_shard_extra_file_rejected(self):
        self.root.chmod(0o700)
        self.file("manifest.json")
        self.file("extra")
        with self.assertRaisesRegex(ValueError, "inventory_changed"):
            m.shard_stat_inventory(self.root, {"shards": []}, "x")

    def test_shard_public_mode_rejected(self):
        self.root.chmod(0o700)
        self.file("manifest.json").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "unsafe_encoded_metadata"):
            m.shard_stat_inventory(self.root, {"shards": []}, "x")

    def test_shard_stat_identity_drift_rejected(self):
        self.root.chmod(0o700)
        self.file("manifest.json")
        with self.assertRaisesRegex(ValueError, "metadata_inventory_changed"):
            m.shard_stat_inventory(self.root, {"shards": []}, "0" * 64)

    def test_report_is_private_and_never_overwritten(self):
        (self.root / "evidence").mkdir()
        m.publish(self.root, m.REPORT, {"status": "fixture"})
        before = (self.root / m.REPORT).read_bytes()
        self.assertEqual((self.root / m.REPORT).stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            m.publish(self.root, m.REPORT, {"status": "changed"})
        self.assertEqual((self.root / m.REPORT).read_bytes(), before)

    def test_output_allowlist(self):
        with self.assertRaisesRegex(ValueError, "output_not_allowlisted"):
            m.publish(self.root, "configs/other.json", {})

    def test_existing_output_stops_before_reaudit(self):
        self.file(m.REPORT)
        with patch.object(m, "audit", side_effect=AssertionError("no reread")):
            with self.assertRaisesRegex(ValueError, "output_already_exists"):
                m.run(self.root, "x")

    def test_edited_plan_boundary_rejected_before_other_reads(self):
        for field, value in (("gpu_authorized", True), ("production_eligible", 0)):
            plan = {
                "schema_version": "lumi.sft-gpu-smoke-readiness-plan.v0.1",
                "authorization_boundary": {**m.BOUNDARY, field: value},
                "cpu_plan_sha256": m.CPU_PLAN_SHA, "bindings": {},
                "launch_draft": m.DRAFT, "pending_prelaunch_gates": m.PENDING,
            }
            raw = m.canonical(plan)
            self.file(m.PLAN, raw)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "boundary"):
                m.audit(self.root, hashlib.sha256(raw).hexdigest())

    def test_verification_rejects_replay_difference(self):
        original = {"status": "fixture", "count": 1}
        original["report_sha256"] = m.digest(original)
        self.file(m.REPORT, m.canonical(original))
        changed = copy.deepcopy(original)
        changed["count"] = 2
        with patch.object(m, "audit", return_value=changed):
            with self.assertRaisesRegex(ValueError, "replayed_report_differs"):
                m.run(self.root, "x", verify=True)
        self.assertFalse((self.root / m.VERIFICATION).exists())

    def synthetic_bound_plan(self):
        """Build a whole metadata lineage from fixtures, never reading the repository."""
        def sealed(field, **extra):
            value = {"status": "PASS_fixture", "production_eligible": False, **extra}
            return {**value, field: m.digest(value)}

        def bind(path, value):
            raw = value if isinstance(value, bytes) else m.canonical(value)
            self.file(path, raw)
            return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

        registry = bind(m.EXTRA_PATHS["source_registry"], {})
        contract = {"source_registry": {"sha256": registry["sha256"]},
                    "learning_candidate": {"scope": "private_research_nonproduction",
                                           "production_sft_allowed": False,
                                           "production_dataset_seal_allowed": False,
                                           "external_distribution_allowed": False}}
        manifest = {"production_eligible": False, "shards": [], "records": 0,
                    "splits": {}, "seal_sha256": "a" * 64}
        manifest["manifest_sha256"] = m.digest(manifest)
        contents = {
            "current_data_contract": contract, "encoded_manifest": manifest,
            "cpu_smoke_recovery_report": sealed("report_sha256"),
            "cpu_smoke_recovery_verification": sealed("verification_sha256"),
            "current_shard_receipt": sealed("receipt_sha256"),
            "current_shard_verification": sealed("verification_sha256"),
            "worker_implementation": (
                b"from .distributed_training import _validate_execution_scope\n"
                b"_validate_execution_scope()\n"
                b"Model.from_pretrained(path, local_files_only=True)\n"
            ),
            "distributed_training_runtime": b"def _validate_paid_authorization(): pass",
        }
        cpu = {"bindings": {name: bind(f"fixtures/{name}", value)
                            for name, value in contents.items()},
               "input": {"manifest_internal_sha256": manifest["manifest_sha256"],
                         "inventory_sha256": m.digest([]), "seal_internal_sha256": "a" * 64,
                         "encoded_directory": "fixture_encoded",
                         "metadata_inventory_sha256": "b" * 64}}
        cpu_hash = bind(m.CPU_PLAN, cpu)["sha256"]
        bindings = {"source_registry": registry}
        bindings["distributed_report"] = bind(
            m.EXTRA_PATHS["distributed_report"], sealed("report_sha256"))
        bindings["distributed_verification"] = bind(
            m.EXTRA_PATHS["distributed_verification"], sealed(
                "verification_sha256", plan_file_sha256=cpu_hash,
                report_file_sha256=bindings["distributed_report"]["sha256"]))
        bindings["final_base"] = bind(m.EXTRA_PATHS["final_base"], {
            "checkpoint": {"step": 76294, "training_tokens": 80_000_000_000}})
        bindings["exporter"] = bind(m.EXTRA_PATHS["exporter"],
                                    b'_write_json(temporary / EXPORT_REPORT_FILENAME, report)')
        for name in ("implementation", "runner", "tests"):
            bindings[name] = bind(m.EXTRA_PATHS[name], b"# synthetic code\n")
        plan = {"schema_version": "lumi.sft-gpu-smoke-readiness-plan.v0.1",
                "authorization_boundary": m.BOUNDARY, "cpu_plan_sha256": cpu_hash,
                "bindings": bindings, "launch_draft": m.DRAFT,
                "pending_prelaunch_gates": m.PENDING}
        return bind(m.PLAN, plan)["sha256"], cpu_hash

    def test_synthetic_end_to_end_report_and_fresh_replay(self):
        plan_hash, cpu_hash = self.synthetic_bound_plan()
        # Real stat-only behavior is covered separately; fake only this fixture inventory.
        with patch.object(m, "CPU_PLAN_SHA", cpu_hash), patch.object(
            m, "shard_stat_inventory", return_value={"files_stat_checked": 31}
        ):
            report = m.run(self.root, plan_hash)
            verification = m.run(self.root, plan_hash, verify=True)
        self.assertFalse(report["launch_ready"])
        self.assertFalse(verification["production_eligible"])
        self.assertTrue(verification["checks"]["full_report_recomputed_exact"])
        self.assertEqual(report["resource_accounting"]["model_weight_bytes_read"], 0)
        self.assertEqual(report["resource_accounting"]["encoded_index_row_reads"], 0)
        self.assertEqual(report["pending_prelaunch_gates"], m.PENDING)


if __name__ == "__main__":
    unittest.main()
