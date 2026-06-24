"""Deployment, release, and serving command handlers for the project CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerospace_prognostics.data.cmapss import CMAPSS_SUBSETS
from aerospace_prognostics.deployment.artifacts import (
    benchmark_cmapss_model_artifact,
    build_cmapss_promotion_report,
    build_cmapss_release_bundle,
    inspect_cmapss_model_artifact,
    load_cmapss_model_artifact,
    save_cmapss_model_artifact,
    train_cmapss_hgb_policy_artifact,
    validate_cmapss_model_artifact,
    write_cmapss_model_card_markdown,
    write_cmapss_promotion_report_markdown,
    write_cmapss_release_bundle_markdown,
)
from aerospace_prognostics.deployment.provenance import (
    build_release_provenance,
    write_release_provenance_json,
    write_release_provenance_markdown,
)
from aerospace_prognostics.deployment.sbom import build_uv_lock_cyclonedx_sbom


def register_deployment_artifact_commands(subparsers: Any) -> None:
    package_hgb = subparsers.add_parser(
        "cmapss-package-hgb-policy",
        help="Train and package the validation-selected HGB policy for deployment",
    )
    package_hgb.add_argument("--data-dir", type=Path, required=True)
    package_hgb.add_argument("--subset", choices=CMAPSS_SUBSETS, required=True)
    package_hgb.add_argument("--output-path", type=Path, required=True)
    package_hgb.add_argument("--metadata-json", type=Path)
    package_hgb.add_argument("--model-card-markdown", type=Path)
    package_hgb.add_argument("--rul-cap", type=int, default=125)
    package_hgb.add_argument("--random-state", type=int, default=42)
    package_hgb.add_argument("--n-regimes", type=int, default=6)
    package_hgb.add_argument("--no-standardize", action="store_true")

    inspect_artifact = subparsers.add_parser(
        "cmapss-inspect-artifact",
        help="Inspect packaged C-MAPSS artifact metadata and serving contract",
    )
    inspect_artifact.add_argument("--model-artifact", type=Path, required=True)
    inspect_artifact.add_argument("--output-json", type=Path)

    predict_artifact = subparsers.add_parser(
        "cmapss-predict-artifact",
        help="Run batch inference with a packaged C-MAPSS model artifact",
    )
    predict_artifact.add_argument("--model-artifact", type=Path, required=True)
    predict_artifact.add_argument("--input-csv", type=Path, required=True)
    predict_artifact.add_argument("--output-json", type=Path)


def register_deployment_release_commands(subparsers: Any) -> None:
    benchmark_artifact = subparsers.add_parser(
        "cmapss-benchmark-artifact",
        help="Benchmark packaged C-MAPSS artifact inference latency",
    )
    benchmark_artifact.add_argument("--model-artifact", type=Path, required=True)
    benchmark_artifact.add_argument("--input-csv", type=Path, required=True)
    benchmark_artifact.add_argument("--runs", type=int, default=20)
    benchmark_artifact.add_argument("--warmup-runs", type=int, default=3)
    benchmark_artifact.add_argument("--max-p95-latency-ms", type=float)
    benchmark_artifact.add_argument("--output-json", type=Path)

    validate_artifact = subparsers.add_parser(
        "cmapss-validate-artifact",
        help="Validate a packaged C-MAPSS model artifact before promotion",
    )
    validate_artifact.add_argument("--model-artifact", type=Path, required=True)
    validate_artifact.add_argument("--metadata-json", type=Path)
    validate_artifact.add_argument("--input-csv", type=Path)
    validate_artifact.add_argument("--output-json", type=Path)

    promotion_report = subparsers.add_parser(
        "cmapss-promotion-report",
        help="Compose deployment gate evidence for C-MAPSS artifact promotion",
    )
    promotion_report.add_argument("--validation-json", type=Path, required=True)
    promotion_report.add_argument("--benchmark-json", type=Path, required=True)
    promotion_report.add_argument("--model-card-markdown", type=Path)
    promotion_report.add_argument("--sbom-json", type=Path)
    promotion_report.add_argument("--output-json", type=Path, required=True)
    promotion_report.add_argument("--output-markdown", type=Path)

    release_bundle = subparsers.add_parser(
        "cmapss-release-bundle",
        help="Compose auditable release-candidate evidence for a C-MAPSS deployment",
    )
    release_bundle.add_argument("--release-name", required=True)
    release_bundle.add_argument("--model-artifact", type=Path, required=True)
    release_bundle.add_argument("--metadata-json", type=Path, required=True)
    release_bundle.add_argument("--model-card-markdown", type=Path, required=True)
    release_bundle.add_argument("--promotion-json", type=Path, required=True)
    release_bundle.add_argument("--sbom-json", type=Path, required=True)
    release_bundle.add_argument("--dashboard-payload-json", type=Path)
    release_bundle.add_argument("--dashboard-html", type=Path)
    release_bundle.add_argument("--container-manifest-json", type=Path)
    release_bundle.add_argument("--container-image-ref")
    release_bundle.add_argument("--output-json", type=Path, required=True)
    release_bundle.add_argument("--output-markdown", type=Path)

    release_provenance = subparsers.add_parser(
        "generate-release-provenance",
        help="Generate in-toto/SLSA-style provenance for a release bundle",
    )
    release_provenance.add_argument("--release-bundle-json", type=Path, required=True)
    release_provenance.add_argument("--repository")
    release_provenance.add_argument("--git-sha")
    release_provenance.add_argument("--git-ref")
    release_provenance.add_argument("--workflow")
    release_provenance.add_argument("--run-id")
    release_provenance.add_argument("--run-attempt")
    release_provenance.add_argument("--actor")
    release_provenance.add_argument("--builder-id")
    release_provenance.add_argument("--output-json", type=Path, required=True)
    release_provenance.add_argument("--output-markdown", type=Path)

    serve_api = subparsers.add_parser(
        "serve-api",
        help="Serve a packaged model artifact with FastAPI",
    )
    serve_api.add_argument("--model-artifact", type=Path, required=True)
    serve_api.add_argument("--model-sha256")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8000)

    sbom = subparsers.add_parser(
        "generate-sbom",
        help="Generate a CycloneDX-style SBOM from uv.lock",
    )
    sbom.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    sbom.add_argument("--output-json", type=Path, required=True)


def handle_deployment_command(args: argparse.Namespace) -> int | None:
    if args.command == "cmapss-package-hgb-policy":
        packaged = train_cmapss_hgb_policy_artifact(
            args.data_dir,
            args.subset,
            rul_cap=args.rul_cap,
            random_state=args.random_state,
            n_regimes=args.n_regimes,
            standardize=not args.no_standardize,
        )
        artifact_path = save_cmapss_model_artifact(packaged.artifact, args.output_path)
        print(f"artifact={artifact_path}")
        print(f"subset={packaged.artifact.subset}")
        print(f"model={packaged.artifact.model_name}")
        print(f"feature_policy={packaged.artifact.feature_policy}")
        print(f"hgb_policy={packaged.artifact.hgb_policy}")
        print(f"artifact_id={packaged.artifact.promotion_metadata['artifact_id']}")
        print(f"rmse={packaged.result.rmse:.6f}")
        print(f"nasa_score={packaged.result.nasa_score:.6f}")
        if args.metadata_json is not None:
            _write_json_payload(
                {
                    "artifact": packaged.artifact.metadata(),
                    "result": packaged.result.to_dict(),
                },
                args.metadata_json,
            )
        if args.model_card_markdown is not None:
            model_card_path = write_cmapss_model_card_markdown(
                packaged.artifact,
                packaged.result,
                args.model_card_markdown,
            )
            print(f"model_card_markdown={model_card_path}")
        return 0

    if args.command == "cmapss-inspect-artifact":
        inspection = inspect_cmapss_model_artifact(args.model_artifact)
        identity = inspection["artifact_identity"]
        model = inspection["model"]
        uncertainty = inspection["uncertainty"]
        input_contract = inspection["input_contract"]
        reference_stats = inspection["reference_stats"]
        checks = inspection["checks"]
        print(f"schema_version={inspection['schema_version']}")
        print(f"artifact_sha256={inspection['artifact_sha256']}")
        print(f"artifact_id={identity.get('artifact_id')}")
        print(f"artifact_schema={identity.get('schema_version')}")
        print(f"dataset={model['dataset']}")
        print(f"subset={model['subset']}")
        print(f"model={model['model_name']}")
        print(f"feature_policy={model['feature_policy']}")
        print(f"hgb_policy={model['hgb_policy']}")
        print(f"input_columns={input_contract['input_column_count']}")
        print(f"feature_columns={input_contract['feature_column_count']}")
        print(f"reference_columns={reference_stats['column_count']}")
        print(f"uncertainty_method={uncertainty.get('method')}")
        print(f"uncertainty_confidence={uncertainty.get('confidence')}")
        check_summary = ",".join(
            f"{name}:{value}" for name, value in sorted(checks.items())
        )
        print(f"checks={check_summary}")
        if args.output_json is not None:
            _write_json_payload(inspection, args.output_json)
            print(f"output_json={args.output_json}")
        return 0

    if args.command == "cmapss-predict-artifact":
        import pandas as pd

        artifact = load_cmapss_model_artifact(args.model_artifact)
        telemetry = pd.read_csv(args.input_csv)
        predictions = artifact.predict_from_frame(telemetry)
        payload = {
            "dataset": artifact.dataset,
            "subset": artifact.subset,
            "model_name": artifact.model_name,
            "rul_cap": artifact.rul_cap,
            "predictions": [prediction.to_dict() for prediction in predictions],
        }
        print(f"model={artifact.model_name}")
        print(f"predictions={len(predictions)}")
        for prediction in predictions:
            print(
                f"unit_number={prediction.unit_number},"
                f"predicted_rul={prediction.predicted_rul:.6f}"
            )
        if args.output_json is not None:
            _write_json_payload(payload, args.output_json)
        return 0

    if args.command == "cmapss-benchmark-artifact":
        benchmark = benchmark_cmapss_model_artifact(
            args.model_artifact,
            args.input_csv,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            max_p95_latency_ms=args.max_p95_latency_ms,
        )
        print(f"status={benchmark.status}")
        print(f"model_size_bytes={benchmark.model_size_bytes}")
        print(f"input_rows={benchmark.input_rows}")
        print(f"prediction_count={benchmark.prediction_count}")
        print(f"latency_p50_ms={benchmark.latency_ms['p50']:.6f}")
        print(f"latency_p95_ms={benchmark.latency_ms['p95']:.6f}")
        for problem in benchmark.problems:
            print(f"problem={problem}")
        if args.output_json is not None:
            _write_json_payload(benchmark.to_dict(), args.output_json)
        return 0 if benchmark.status == "ok" else 1

    if args.command == "cmapss-validate-artifact":
        validation = validate_cmapss_model_artifact(
            args.model_artifact,
            metadata_json=args.metadata_json,
            input_csv=args.input_csv,
        )
        print(f"status={validation.status}")
        artifact_id = validation.artifact_identity.get("artifact_id")
        if artifact_id:
            print(f"artifact_id={artifact_id}")
        if validation.prediction_count is not None:
            print(f"prediction_count={validation.prediction_count}")
        for problem in validation.problems:
            print(f"problem={problem}")
        if args.output_json is not None:
            _write_json_payload(validation.to_dict(), args.output_json)
        return 0 if validation.status == "ok" else 1

    if args.command == "cmapss-promotion-report":
        report = build_cmapss_promotion_report(
            args.validation_json,
            args.benchmark_json,
            model_card_markdown=args.model_card_markdown,
            sbom_json=args.sbom_json,
        )
        print(f"status={report.status}")
        artifact_id = report.artifact_identity.get("artifact_id")
        if artifact_id:
            print(f"artifact_id={artifact_id}")
        print(f"gates_passed={sum(report.gates.values())}")
        print(f"gates_total={len(report.gates)}")
        for problem in report.problems:
            print(f"problem={problem}")
        _write_json_payload(report.to_dict(), args.output_json)
        if args.output_markdown is not None:
            markdown_path = write_cmapss_promotion_report_markdown(
                report,
                args.output_markdown,
            )
            print(f"output_markdown={markdown_path}")
        return 0 if report.status == "ok" else 1

    if args.command == "cmapss-release-bundle":
        bundle = build_cmapss_release_bundle(
            release_name=args.release_name,
            model_artifact=args.model_artifact,
            metadata_json=args.metadata_json,
            model_card_markdown=args.model_card_markdown,
            promotion_json=args.promotion_json,
            sbom_json=args.sbom_json,
            dashboard_payload_json=args.dashboard_payload_json,
            dashboard_html=args.dashboard_html,
            container_manifest_json=args.container_manifest_json,
            container_image_ref=args.container_image_ref,
        )
        print(f"status={bundle.status}")
        print(f"release_name={bundle.release_name}")
        artifact_id = bundle.artifact_identity.get("artifact_id")
        if artifact_id:
            print(f"artifact_id={artifact_id}")
        if bundle.container_image_ref:
            print(f"container_image_ref={bundle.container_image_ref}")
        print(f"gates_passed={sum(bundle.gates.values())}")
        print(f"gates_total={len(bundle.gates)}")
        for problem in bundle.problems:
            print(f"problem={problem}")
        _write_json_payload(bundle.to_dict(), args.output_json)
        if args.output_markdown is not None:
            markdown_path = write_cmapss_release_bundle_markdown(
                bundle,
                args.output_markdown,
            )
            print(f"output_markdown={markdown_path}")
        return 0 if bundle.status == "ok" else 1

    if args.command == "generate-release-provenance":
        provenance = build_release_provenance(
            args.release_bundle_json,
            repository=args.repository,
            git_sha=args.git_sha,
            git_ref=args.git_ref,
            workflow=args.workflow,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            actor=args.actor,
            builder_id=args.builder_id,
        )
        print(f"status={provenance.status}")
        print(f"release_name={provenance.release_name}")
        print(f"subject_count={provenance.summary['subject_count']}")
        git_sha = provenance.summary.get("git_sha")
        if git_sha:
            print(f"git_sha={git_sha}")
        for problem in provenance.problems:
            print(f"problem={problem}")
        json_path = write_release_provenance_json(provenance, args.output_json)
        print(f"output_json={json_path}")
        if args.output_markdown is not None:
            markdown_path = write_release_provenance_markdown(
                provenance,
                args.output_markdown,
            )
            print(f"output_markdown={markdown_path}")
        return 0 if provenance.status == "ok" else 1

    if args.command == "serve-api":
        import uvicorn

        from aerospace_prognostics.serving.api import create_app

        uvicorn.run(
            create_app(args.model_artifact, expected_artifact_sha256=args.model_sha256),
            host=args.host,
            port=args.port,
        )
        return 0

    if args.command == "generate-sbom":
        sbom = build_uv_lock_cyclonedx_sbom(args.lockfile)
        _write_json_payload(sbom, args.output_json)
        print(f"sbom_json={args.output_json}")
        print(f"spec_version={sbom['specVersion']}")
        print(f"component_count={len(sbom['components'])}")
        return 0

    return None


def _write_json_payload(payload: object, path: Path) -> None:
    output_path = _prepare_output_path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
