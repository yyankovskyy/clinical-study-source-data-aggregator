from pathlib import Path
import pandas as pd
from clinical_aggregator.adam_engine import derive_vars_dy
from clinical_aggregator.agent import AgentRequest, ClinicalStudyAgent


def test_no_study_day_zero() -> None:
    df = pd.DataFrame({
        "TRTSDT": ["2024-01-10", "2024-01-10"],
        "ADT": ["2024-01-09", "2024-01-10"],
    })
    res = derive_vars_dy(df, "TRTSDT", "ADT", "ADY")
    assert res["ADY"].tolist() == [-1.0, 1.0]


def test_agent_prompt_validation() -> None:
    req, missing = ClinicalStudyAgent.parse_natural_language_prompt("Build ADaM tables")
    assert req is None
    assert len(missing) == 3

    req_ok, miss_ok = ClinicalStudyAgent.parse_natural_language_prompt(
        "Read https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01 "
        "build ADSL ADVS ADLB into ./test_out"
    )
    assert req_ok is not None
    assert miss_ok == []
    assert "ADSL" in req_ok.target_adam_tables


def test_end_to_end_synthetic_sdtm_to_output_folder(tmp_path: Path) -> None:
    sdtm_dir = tmp_path / "sdtm_src"
    out_dir = tmp_path / "custom_output"
    sdtm_dir.mkdir()

    pd.DataFrame({
        "STUDYID": ["STUDY01"] * 4,
        "DOMAIN": ["DM"] * 4,
        "USUBJID": ["S01", "S02", "S03", "S04"],
        "SUBJID": ["01", "02", "03", "04"],
        "SITEID": ["101", "101", "102", "102"],
        "AGE": [62, 71, 78, 84],
        "SEX": ["M", "F", "M", "F"],
        "RACE": ["WHITE"] * 4,
        "ARMCD": ["PBO", "XAN", "PBO", "XAN"],
        "ARM": ["Placebo", "Xanomeline High Dose", "Placebo", "Xanomeline High Dose"],
        "RFSTDTC": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "RFENDTC": ["2024-06-01", "2024-03-01", "2024-06-03", "2024-02-15"],
    }).to_csv(sdtm_dir / "dm.csv", index=False)

    pd.DataFrame({
        "STUDYID": ["STUDY01"] * 4, "DOMAIN": ["EX"] * 4,
        "USUBJID": ["S01", "S02", "S03", "S04"], "EXSEQ": [1, 1, 1, 1],
        "EXTRT": ["PLACEBO", "XANOMELINE", "PLACEBO", "XANOMELINE"],
        "EXDOSE": [0, 81, 0, 81],
        "EXSTDTC": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "EXENDTC": ["2024-06-01", "2024-03-01", "2024-06-03", "2024-02-15"],
    }).to_csv(sdtm_dir / "ex.csv", index=False)

    pd.DataFrame({
        "STUDYID": ["STUDY01"] * 4, "DOMAIN": ["DS"] * 4,
        "USUBJID": ["S01", "S02", "S03", "S04"], "DSSEQ": [1, 1, 1, 1],
        "DSTERM": ["COMPLETED", "ADVERSE EVENT", "COMPLETED", "LACK OF EFFICACY"],
        "DSDECOD": ["COMPLETED", "ADVERSE EVENT", "COMPLETED", "LACK OF EFFICACY"],
        "DSCAT": ["DISPOSITION EVENT"] * 4,
        "DSSTDTC": ["2024-06-01", "2024-03-01", "2024-06-03", "2024-02-15"],
    }).to_csv(sdtm_dir / "ds.csv", index=False)

    pd.DataFrame({
        "STUDYID": ["STUDY01"] * 4, "DOMAIN": ["VS"] * 4,
        "USUBJID": ["S01", "S01", "S02", "S02"], "VSSEQ": [1, 2, 1, 2],
        "VSTESTCD": ["SYSBP", "SYSBP", "SYSBP", "SYSBP"],
        "VSTEST": ["Systolic Blood Pressure"] * 4,
        "VSSTRESN": [120.0, 115.0, 135.0, 142.0],
        "VSSTRESC": ["120", "115", "135", "142"],
        "VSSTRESU": ["mmHg"] * 4,
        "VISIT": ["BASELINE", "WEEK 4", "BASELINE", "WEEK 4"],
        "VISITNUM": [1, 4, 1, 4],
        "VSDTC": ["2024-01-01", "2024-01-29", "2024-01-02", "2024-01-30"],
    }).to_csv(sdtm_dir / "vs.csv", index=False)

    pd.DataFrame({
        "STUDYID": ["STUDY01"] * 4, "DOMAIN": ["LB"] * 4,
        "USUBJID": ["S01", "S01", "S02", "S02"], "LBSEQ": [1, 2, 1, 2],
        "LBTESTCD": ["ALT", "ALT", "ALT", "ALT"],
        "LBTEST": ["Alanine Aminotransferase"] * 4,
        "LBSTRESN": [25.0, 30.0, 28.0, 55.0],
        "LBSTRESC": ["25", "30", "28", "55"],
        "LBSTRESU": ["U/L"] * 4,
        "LBSTNRLO": [10.0] * 4, "LBSTNRHI": [40.0] * 4,
        "VISIT": ["BASELINE", "WEEK 4", "BASELINE", "WEEK 4"],
        "VISITNUM": [1, 4, 1, 4],
        "LBDTC": ["2024-01-01", "2024-01-29", "2024-01-02", "2024-01-30"],
    }).to_csv(sdtm_dir / "lb.csv", index=False)

    agent = ClinicalStudyAgent()
    req = AgentRequest(
        source_sdtm_link=str(sdtm_dir),
        target_adam_tables=["ADSL", "ADVS", "ADLB", "ADEG", "ADBCVA"],
        output_location=str(out_dir),
    )
    res = agent.execute_request(req)
    assert (out_dir / "adam_csv" / "adsl.csv").exists()
    assert (out_dir / "adam_csv" / "advs.csv").exists()
    assert (out_dir / "adam_csv" / "adlb.csv").exists()
    assert "UNAVAILABLE" in res["adam_status"]["ADEG"]
    assert "UNAVAILABLE" in res["adam_status"]["ADBCVA"]
    assert res["dropout_results"]["leakage_audit"]["leakage_check_passed"] is True
