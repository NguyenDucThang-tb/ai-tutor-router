from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import socket
import urllib.request
from datetime import UTC, datetime

import pytest

from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from experiments.authorization import (
    approval_context,
    approval_signature_payload,
    context_digest,
    verify_approval_artifact,
    write_not_approved_artifact,
)
from experiments.config import BenchmarkConfig
from experiments.model_catalog import resolve_model
from experiments.provider_verify import main, make_report


def _setup():
    row = DatasetRecord(
        sample_id="fixture-1",
        dataset="gsm8k",
        dataset_revision="fixture-r1",
        original_split="train",
        assigned_split="validation",
        question="2+2?",
        reference_answer="4",
        task_type="word_problem",
        source_id="1",
        source_hash="a" * 64,
    )
    manifest = DatasetManifest(
        dataset="gsm8k",
        source_organization="fixture",
        source_url="https://example.invalid/data",
        revision="fixture-r1",
        license="fixture",
        access_note="fixture",
        downloaded_at_utc=datetime.now(UTC),
        raw_files={},
        transformation_history=["fixture"],
        seed=42,
        train_fraction=0.9,
        near_duplicate_threshold=0.9,
        assigned_split_counts={"validation": 1},
        dataset_content_sha256=sha256_records([row]),
        processed_file_sha256={},
    )
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
        max_samples=1,
        max_requests=2,
        max_run_budget_usd=2,
        per_request_budget_usd=0.1,
        allow_real_provider=True,
        experiment_approval_id="fixture-approval",
    )
    return config, row, manifest


def test_offline_provider_verification_makes_no_network_calls(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: real\nprovider: litellm\n"
        "small_model_identifier: gemini/gemini-3.5-flash-lite\n"
        "strong_model_identifier: gemini/gemini-3.8-flash\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("offline verification attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    report = make_report(config_path)
    assert report["network_calls"] == 0
    assert report["secrets_read"] is False
    assert report["statuses"]["PROVIDER_DOCUMENTED"] is True
    assert report["statuses"]["ACCOUNT_ACCESS_VERIFIED"] is False
    assert report["statuses"]["PAID_EXECUTION_AUTHORIZED"] is False
    assert main(["--config", str(config_path), "--offline"]) == 0
    assert json.loads(capsys.readouterr().out)["network_calls"] == 0


def test_unknown_and_deprecated_models_are_not_documented_as_current(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: real\nprovider: litellm\n"
        "small_model_identifier: openai/gpt-4.1-mini-2025-04-14\n"
        "strong_model_identifier: openai/gpt-6.2-sol\n"
    )
    report = make_report(config_path)
    assert report["statuses"]["PROVIDER_DOCUMENTED"] is False
    assert report["models"][0]["status"] == "UNSUPPORTED_OR_STALE"
    assert resolve_model("openai/gpt-4.1-mini-2025-04-14").status == "deprecated"

    config_path.write_text(
        "mode: real\nprovider: litellm\n"
        "small_model_identifier: provider/unknown-model\n"
        "strong_model_identifier: openai/gpt-6.2-sol\n"
    )
    assert make_report(config_path)["statuses"]["PROVIDER_DOCUMENTED"] is False


def test_offline_provider_report_marks_stale_price_metadata(tmp_path, monkeypatch):
    from datetime import date

    from experiments import provider_verify

    class FutureDate:
        @staticmethod
        def today():
            return date(2026, 12, 1)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: real\nprovider: litellm\n"
        "small_model_identifier: gemini/gemini-3.5-flash-lite\n"
        "strong_model_identifier: gemini/gemini-3.8-flash\n"
    )
    monkeypatch.setattr(provider_verify, "date", FutureDate)
    report = provider_verify.make_report(config_path)
    assert report["statuses"]["PROVIDER_DOCUMENTED"] is False
    assert all(not model["pricing_current"] for model in report["models"])


def test_account_verification_mode_is_explicitly_unsupported(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: mock\nprovider: mock\nsmall_model_identifier: a\nstrong_model_identifier: b\n"
    )
    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(config_path)])


def test_locked_litellm_catalog_has_exact_gemini_chat_mappings():
    distribution = importlib.metadata.distribution("litellm")
    assert distribution.version == "1.110.2"
    catalog_path = distribution.locate_file("litellm/model_prices_and_context_window_backup.json")
    catalog = json.loads(catalog_path.read_text("utf-8"))
    for model_id, input_price, output_price, max_input, max_output in (
        ("gemini/gemini-3.5-flash-lite", 3e-7, 2.5e-6, 1_048_576, 65_536),
        ("gemini/gemini-3.8-flash", 7.5e-7, 3.75e-6, 1_048_576, 65_536),
    ):
        item = catalog[model_id]
        assert item["litellm_provider"] == "gemini"
        assert item["mode"] == "chat"
        assert "/v1/chat/completions" in item["supported_endpoints"]
        assert item["input_cost_per_token"] == input_price
        assert item["output_cost_per_token"] == output_price
        assert item["max_input_tokens"] == max_input
        assert item["max_output_tokens"] == max_output


def test_unsigned_approval_draft_cannot_authorize_execution(tmp_path):
    config, row, manifest = _setup()
    path = tmp_path / "approval.json"
    context = approval_context(config, [row], manifest, "validation", None)
    write_not_approved_artifact(path, context)
    valid, reason = verify_approval_artifact(
        path,
        config=config,
        records=[row],
        manifest=manifest,
        split="validation",
        collection_plan=None,
        signing_key="offline-test-signing-key-32-bytes",
    )
    assert not valid
    assert "not APPROVED" in reason


def test_approval_is_bound_to_config_and_signature(tmp_path):
    config, row, manifest = _setup()
    context = approval_context(config, [row], manifest, "validation", None)
    artifact = {
        "schema_version": "paid-approval-v1",
        "immutable_config": context,
        "immutable_config_sha256": context_digest(context),
        "approval": {
            "approval_id": "fixture-approval",
            "approved_by": "human-operator",
            "approval_timestamp": "2026-11-22T00:00:00Z",
            "status": "APPROVED",
            "signature": "",
        },
    }
    key = "offline-test-signing-key-32-bytes"
    artifact["approval"]["signature"] = hmac.new(
        key.encode(), approval_signature_payload(artifact), hashlib.sha256
    ).hexdigest()
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert verify_approval_artifact(
        path,
        config=config,
        records=[row],
        manifest=manifest,
        split="validation",
        collection_plan=None,
        signing_key=key,
    )[0]

    changed = config.model_copy(update={"max_run_budget_usd": 3})
    valid, reason = verify_approval_artifact(
        path,
        config=changed,
        records=[row],
        manifest=manifest,
        split="validation",
        collection_plan=None,
        signing_key=key,
    )
    assert not valid
    assert "does not match this run" in reason

    for changed_config, changed_split, changed_rows, changed_plan in (
        (
            config.model_copy(update={"strong_model_identifier": "other/model"}),
            "validation",
            [row],
            None,
        ),
        (config, "train", [row], None),
        (config, "validation", [], None),
        (config, "validation", [row], {"plan_sha256": "b" * 64}),
    ):
        valid, reason = verify_approval_artifact(
            path,
            config=changed_config,
            records=changed_rows,
            manifest=manifest,
            split=changed_split,
            collection_plan=changed_plan,
            signing_key=key,
        )
        assert not valid
        assert "does not match this run" in reason

    artifact["approval"]["approved_by"] = "tampered"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    valid, reason = verify_approval_artifact(
        path,
        config=config,
        records=[row],
        manifest=manifest,
        split="validation",
        collection_plan=None,
        signing_key=key,
    )
    assert not valid
    assert "signature is invalid" in reason
