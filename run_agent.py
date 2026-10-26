#!/usr/bin/env python3
import argparse
from clinical_aggregator.agent import AgentRequest, ClinicalStudyAgent
from clinical_aggregator.ingestion import PHUSE_PILOT01_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical Study Source Data Aggregator AI Agent")
    parser.add_argument("--prompt", type=str, help="Natural language prompt containing source link, ADaM tables, and output folder")
    parser.add_argument("--source", type=str, help="SDTM source GitHub URL, ZIP file, or local directory")
    parser.add_argument("--tables", nargs="+", help="Target ADaM tables (e.g., ADSL ADVS ADLB ADEG ADBCVA)")
    parser.add_argument("--output-dir", type=str, help="Destination local directory for generated ADaM tables and reports")
    args = parser.parse_args()

    agent = ClinicalStudyAgent()

    if args.prompt:
        req, missing = agent.parse_natural_language_prompt(args.prompt)
        if missing:
            print("The AI Agent requires the following missing parameter(s):")
            for m in missing:
                print(f"  - {m}")
            return
    else:
        source = args.source or input(f"1. Enter SDTM Source Link/Path [{PHUSE_PILOT01_URL}]: ").strip() or PHUSE_PILOT01_URL
        tables_raw = " ".join(args.tables) if args.tables else (input("2. Enter Required ADaM Tables [ADSL ADVS ADLB]: ").strip() or "ADSL ADVS ADLB")
        tables = [t.strip().upper() for t in tables_raw.replace(",", " ").split() if t.strip()]
        out_dir = args.output_dir or input("3. Enter Output Location Folder [./agent_outputs]: ").strip() or "./agent_outputs"
        req = AgentRequest(source_sdtm_link=source, target_adam_tables=tables, output_location=out_dir)

    print("\n[Agent] Running SDTM-to-ADaM Aggregation...")
    print(f"  -> Source SDTM:  {req.source_sdtm_link}")
    print(f"  -> ADaM Tables:  {req.target_adam_tables}")
    print(f"  -> Output Dir:   {req.output_location}\n")

    res = agent.execute_request(req)
    print("=== EXECUTION COMPLETE ===")
    for k, v in res["adam_status"].items():
        print(f"  * {k}: {v}")
    print(f"  * Source Immutability Verified: {res['source_immutability_verified']}")
    print(f"  * Data Leakage Check Passed:    {res['dropout_results']['leakage_audit']['leakage_check_passed']}")
    print(f"  * All Tables & Reports Saved:   {res['output_location']}")
    print(f"  * Deliverable ZIP Archive:      {res['zip_bundle_path']}")


if __name__ == "__main__":
    main()
