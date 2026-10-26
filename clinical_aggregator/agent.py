import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

from clinical_aggregator.adam_engine import LineageManifest, build_adsl, build_bds_datasets
from clinical_aggregator.analytics_dropout import compute_missingness_report, run_dropout_and_survival_pipeline
from clinical_aggregator.ingestion import PHUSE_PILOT01_URL, ingest_sdtm_source, verify_source_immutability
from clinical_aggregator.validation import validate_sdtm

load_dotenv()


@dataclass
class AgentRequest:
    source_sdtm_link: str = PHUSE_PILOT01_URL
    target_adam_tables: List[str] = field(default_factory=lambda: ["ADSL", "ADVS", "ADLB"])
    output_location: str = "./agent_outputs"
    approve_partial_date_imputation: bool = False


class ClinicalStudyAgent:
    def __init__(self, model_name: str = "gemini-4-argon") -> None:
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def ask_gemini_argon(self, prompt: str, system_instruction: str = "") -> str:
        if not self.api_key:
            return (
                "*(Offline Deterministic Mode - GEMINI_API_KEY not set in .env)*\n"
                "All numerical transformations, validations, survival curves, and leakage checks "
                "were executed deterministically by the Python CDISC engine."
            )
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            if hasattr(client, "interactions"):
                interaction = client.interactions.create(
                    model=self.model_name,
                    input=prompt,
                    system_instruction=system_instruction or "You are a senior CDISC SDTM/ADaM biostatistician.",
                )
                return getattr(interaction, "output_text", str(interaction))
            resp = client.models.generate_content(model=self.model_name, contents=prompt)
            return resp.text or ""
        except Exception as exc:
            return f"*(Gemini API note: {exc} - deterministic Python pipeline outputs unaffected.)*"

    @staticmethod
    def parse_natural_language_prompt(user_prompt: str) -> Tuple[Optional[AgentRequest], List[str]]:
        missing = []
        url_match = re.search(r"(https?://[^\s,]+)", user_prompt)
        source_link = url_match.group(1) if url_match else ""

        tables = [t for t in ["ADSL", "ADVS", "ADLB", "ADEG", "ADBCVA"] if re.search(rf"\b{t}\b", user_prompt.upper())]
        out_match = re.search(r"(?:output|to|into|folder|directory|location)\s*[:=]?\s*([./~a-zA-Z0-9_-]+/[a-zA-Z0-9_./-]+|\./[a-zA-Z0-9_./-]+)", user_prompt, re.IGNORECASE)
        out_loc = out_match.group(1) if out_match else ""

        if not source_link:
            missing.append("1. `source_sdtm_link` (e.g., https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01)")
        if not tables:
            missing.append("2. `target_adam_tables` (e.g., ADSL, ADVS, ADLB, ADEG, ADBCVA)")
        if not out_loc:
            missing.append("3. `output_location` (e.g., ./my_study_outputs)")

        if missing:
            return None, missing
        return AgentRequest(source_sdtm_link=source_link, target_adam_tables=tables, output_location=out_loc), []

    def execute_request(self, req: AgentRequest) -> Dict[str, Any]:
        out_root = Path(req.output_location).expanduser().resolve()
        csv_dir = out_root / "adam_csv"
        rep_dir = out_root / "reports"
        plots_dir = out_root / "plots"
        for d in (csv_dir, rep_dir, plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        bundle = ingest_sdtm_source(req.source_sdtm_link)
        pd.DataFrame(bundle.inventory_summary).to_csv(rep_dir / "sdtm_inventory.csv", index=False)

        val_report = validate_sdtm(bundle)
        val_df = val_report.to_dataframe()
        val_df.to_csv(rep_dir / "sdtm_validation_report.csv", index=False)
        if not val_report.can_proceed:
            raise RuntimeError(f"Pipeline halted due to {val_report.critical_count} CRITICAL SDTM validation error(s).")

        manifest = LineageManifest()
        requested = [t.upper().strip() for t in req.target_adam_tables]
        if "ADSL" not in requested:
            requested.insert(0, "ADSL")

        adsl = build_adsl(bundle, manifest, impute_dates=req.approve_partial_date_imputation)
        adam_datasets, adam_status = build_bds_datasets(bundle, adsl, requested, manifest)

        var_meta_rows = []
        for ds_name, df in adam_datasets.items():
            csv_path = csv_dir / f"{ds_name.lower()}.csv"
            df.to_csv(csv_path, index=False)
            for col in df.columns:
                var_meta_rows.append({
                    "Dataset": ds_name, "Variable": col, "DataType": str(df[col].dtype),
                    "NonNull_Count": int(df[col].notna().sum()), "Null_Count": int(df[col].isna().sum()),
                })

        pd.DataFrame(var_meta_rows).to_csv(rep_dir / "dataset_variable_metadata.csv", index=False)
        manifest.to_dataframe().to_csv(rep_dir / "derivation_lineage_manifest.csv", index=False)
        with open(rep_dir / "adam_transformation_status.json", "w") as f:
            json.dump(adam_status, f, indent=2)

        miss_df = compute_missingness_report(adsl, adam_datasets)
        miss_df.to_csv(rep_dir / "missingness_report.csv", index=False)

        dropout_res = run_dropout_and_survival_pipeline(adsl, adam_datasets, plots_dir)
        dropout_res["by_arm"].to_csv(rep_dir / "discontinuation_by_arm.csv", index=False)
        dropout_res["reasons"].to_csv(rep_dir / "discontinuation_reasons.csv", index=False)
        dropout_res["km_df"].to_csv(rep_dir / "survival_kaplan_meier.csv", index=False)
        dropout_res["cox_df"].to_csv(rep_dir / "survival_cox_ph.csv", index=False)
        dropout_res["cif_df"].to_csv(rep_dir / "competing_risks_cif.csv", index=False)
        with open(rep_dir / "data_leakage_audit.json", "w") as f:
            json.dump(dropout_res["leakage_audit"], f, indent=2)

        immut_ok, _ = verify_source_immutability(bundle.working_dir, bundle.file_hashes_sha256)

        summary_prompt = (
            f"Summarize the clinical study aggregation for Study {val_report.study_id}:\n"
            f"- Source: {req.source_sdtm_link}\n"
            f"- Output Location: {out_root}\n"
            f"- ADaM Status: {adam_status}\n"
            f"- Validation: {val_report.error_count} errors, {val_report.warning_count} warnings\n"
            f"- Discontinuation by arm: {dropout_res['by_arm'].to_dict(orient='records')}\n"
            f"- Leakage Check Passed: {dropout_res['leakage_audit']['leakage_check_passed']}"
        )
        ai_narrative = self.ask_gemini_argon(summary_prompt)
        exec_md = (
            f"# Executive Clinical Data Quality & ADaM Transformation Report\n\n"
            f"- **Study ID**: `{val_report.study_id}`\n"
            f"- **Source SDTM Link**: `{req.source_sdtm_link}`\n"
            f"- **Output Directory**: `{out_root}`\n"
            f"- **Source Immutability Verified (SHA-256)**: `{immut_ok}`\n"
            f"- **ADaM Datasets Status**: `{json.dumps(adam_status)}`\n\n"
            f"## Gemini 4 Argon Clinical Synthesis\n{ai_narrative}\n"
        )
        (rep_dir / "agent_executive_summary.md").write_text(exec_md, encoding="utf-8")

        zip_bundle_path = out_root / f"{val_report.study_id.lower()}_deliverables.zip"
        with zipfile.ZipFile(zip_bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in (csv_dir, rep_dir, plots_dir):
                for fp in folder.rglob("*"):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(out_root))

        return {
            "study_id": val_report.study_id,
            "output_location": str(out_root),
            "zip_bundle_path": str(zip_bundle_path),
            "source_immutability_verified": immut_ok,
            "validation_report": val_report,
            "adam_datasets": adam_datasets,
            "adam_status": adam_status,
            "lineage_manifest": manifest.to_dataframe(),
            "missingness_df": miss_df,
            "dropout_results": dropout_res,
            "ai_narrative": ai_narrative,
        }
