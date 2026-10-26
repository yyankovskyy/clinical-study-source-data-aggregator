from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from clinical_aggregator.ingestion import IngestionBundle, combine_supp

ADVS_LOOKUP = pd.DataFrame([
    {"PARAMCD": "HEIGHT", "PARAM": "Height (cm)", "PARAMN": 1, "PARCAT1": "Subject Characteristic", "ANRLO": np.nan, "ANRHI": np.nan},
    {"PARAMCD": "WEIGHT", "PARAM": "Weight (kg)", "PARAMN": 2, "PARCAT1": "Subject Characteristic", "ANRLO": np.nan, "ANRHI": np.nan},
    {"PARAMCD": "DIABP", "PARAM": "Diastolic Blood Pressure (mmHg)", "PARAMN": 3, "PARCAT1": "Vital Sign", "ANRLO": 60.0, "ANRHI": 80.0},
    {"PARAMCD": "MAP", "PARAM": "Mean Arterial Pressure (mmHg)", "PARAMN": 4, "PARCAT1": "Vital Sign", "ANRLO": 70.0, "ANRHI": 100.0},
    {"PARAMCD": "PULSE", "PARAM": "Pulse Rate (beats/min)", "PARAMN": 5, "PARCAT1": "Vital Sign", "ANRLO": 60.0, "ANRHI": 100.0},
    {"PARAMCD": "SYSBP", "PARAM": "Systolic Blood Pressure (mmHg)", "PARAMN": 6, "PARCAT1": "Vital Sign", "ANRLO": 90.0, "ANRHI": 120.0},
    {"PARAMCD": "TEMP", "PARAM": "Temperature (C)", "PARAMN": 7, "PARCAT1": "Vital Sign", "ANRLO": 36.1, "ANRHI": 37.2},
    {"PARAMCD": "BMI", "PARAM": "Body Mass Index (kg/m^2)", "PARAMN": 8, "PARCAT1": "Subject Characteristic", "ANRLO": 18.5, "ANRHI": 24.9},
    {"PARAMCD": "BSA", "PARAM": "Body Surface Area (m^2)", "PARAMN": 9, "PARCAT1": "Subject Characteristic", "ANRLO": np.nan, "ANRHI": np.nan},
])


@dataclass
class LineageStep:
    step_id: str
    dataset: str
    variables: str
    source_domains: str
    rule_name: str
    admiral_equivalent: str
    rows_affected: int


class LineageManifest:
    def __init__(self) -> None:
        self.steps: List[LineageStep] = []
        self.cardinality_checks: List[Dict[str, Any]] = []
        self.zero_denominators: List[Dict[str, Any]] = []

    def log(self, ds: str, vars_list: List[str], srcs: List[str], rule: str, adm: str, rows: int) -> None:
        sid = f"{ds}-{len(self.steps) + 1:02d}"
        self.steps.append(LineageStep(sid, ds, ", ".join(vars_list), ", ".join(srcs), rule, adm, int(rows)))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.steps]) if self.steps else pd.DataFrame()


def convert_blanks_to_na(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].apply(
                lambda v: np.nan if (v is None or (isinstance(v, str) and v.strip() in ("", "."))) else (v.strip() if isinstance(v, str) else v)
            )
    return out


def derive_vars_dt(df: pd.DataFrame, dtc: str, prefix: str = "A", impute_month: bool = False) -> pd.DataFrame:
    out = df.copy()
    dt_col, dtf_col = f"{prefix}DT", f"{prefix}DTF"
    if dtc not in out.columns:
        out[dt_col], out[dtf_col] = pd.NaT, np.nan
        return out
    dates, flags = [], []
    for val in out[dtc].fillna("").astype(str).str.strip():
        p = val.split("T")[0].strip().split("-")
        if len(p) == 3 and len(p[0]) == 4 and p[1].isdigit() and p[2].isdigit():
            try:
                dates.append(pd.Timestamp(f"{p[0]}-{p[1]}-{p[2]}"))
                flags.append(np.nan)
            except Exception:
                dates.append(pd.NaT)
                flags.append(np.nan)
        elif impute_month and len(p) == 2 and len(p[0]) == 4 and p[1].isdigit():
            dates.append(pd.Timestamp(f"{p[0]}-{p[1]}-01"))
            flags.append("D")
        else:
            dates.append(pd.NaT)
            flags.append(np.nan)
    out[dt_col] = pd.to_datetime(dates)
    out[dtf_col] = pd.Series(flags, index=out.index, dtype="object")
    return out


def derive_vars_dy(df: pd.DataFrame, ref_col: str = "TRTSDT", dt_col: str = "ADT", dy_col: str = "ADY") -> pd.DataFrame:
    out = df.copy()
    ref = pd.to_datetime(out[ref_col], errors="coerce").dt.normalize()
    dt = pd.to_datetime(out[dt_col], errors="coerce").dt.normalize()
    diff = (dt - ref).dt.days
    out[dy_col] = pd.Series(np.where(diff.isna(), np.nan, np.where(diff >= 0, diff + 1, diff)), index=out.index, dtype="float64")
    return out


def derive_vars_merged(
    df: pd.DataFrame, add_df: pd.DataFrame, by_vars: List[str], new_vars: Dict[str, str],
    filter_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    order: Optional[List[str]] = None, mode: str = "first",
    manifest: Optional[LineageManifest] = None, ds_name: str = "ADaM",
) -> pd.DataFrame:
    out = df.copy()
    n0 = len(out)
    sub = add_df[filter_fn(add_df)].copy() if filter_fn is not None else add_df.copy()
    if sub.empty:
        for k in new_vars:
            out[k] = np.nan
        return out
    if order:
        vo = [c for c in order if c in sub.columns]
        if vo:
            sub = sub.sort_values(by=by_vars + vo, na_position="last")
    dedup = sub.drop_duplicates(subset=by_vars, keep="last" if mode == "last" else "first").copy()
    for tcol, scol in new_vars.items():
        if scol in dedup.columns:
            dedup[tcol] = dedup[scol]
    for tcol in new_vars:
        if tcol in out.columns and tcol not in by_vars:
            out = out.drop(columns=[tcol])
    m = out.merge(dedup[by_vars + [t for t in new_vars if t in dedup.columns]], on=by_vars, how="left")
    if len(m) != n0:
        raise RuntimeError(f"Join cardinality violation in {ds_name}: {n0} -> {len(m)}")
    if manifest:
        manifest.cardinality_checks.append({"dataset": ds_name, "vars": list(new_vars.keys()), "rows_before": n0, "rows_after": len(m), "passed": True})
    return m


def build_adsl(bundle: IngestionBundle, manifest: LineageManifest, impute_dates: bool = False) -> pd.DataFrame:
    dm_raw = bundle.get_domain("DM")
    if dm_raw is None or dm_raw.empty:
        raise ValueError("DM domain is required to construct ADSL.")
    dm_comb, supp_cols = combine_supp(dm_raw, bundle.get_domain("SUPPDM"), "DM")
    adsl = convert_blanks_to_na(dm_comb).drop(columns=["DOMAIN"], errors="ignore")
    adsl["SRCDOM"], adsl["SRCROWID"] = "DM", adsl["USUBJID"].astype(str)

    adsl["TRT01P"] = adsl["ARM"]
    adsl["TRT01A"] = adsl["ACTARM"] if "ACTARM" in adsl.columns else adsl["ARM"]
    amap = {"PLACEBO": 0, "XANOMELINE LOW DOSE": 54, "XANOMELINE HIGH DOSE": 81, "SCREEN FAILURE": 99}
    adsl["TRT01PN"] = adsl["TRT01P"].astype(str).str.upper().map(amap).fillna(1)
    adsl["TRT01AN"] = adsl["TRT01PN"]

    ex_raw = bundle.get_domain("EX")
    if ex_raw is not None and not ex_raw.empty:
        ex = convert_blanks_to_na(ex_raw)
        ex["DOSE_N"] = pd.to_numeric(ex["EXDOSE"], errors="coerce")
        ex = derive_vars_dt(derive_vars_dt(ex, "EXSTDTC", "EXST", impute_dates), "EXENDTC", "EXEN", impute_dates)
        vex = lambda d: (d["DOSE_N"] > 0) | ((d["DOSE_N"] == 0) & d["EXTRT"].astype(str).str.upper().str.contains("PLACEBO", na=False))
        adsl = derive_vars_merged(adsl, ex, ["STUDYID", "USUBJID"], {"TRTSDT": "EXSTDT", "TRTSDTF": "EXSTDTF", "TRTSSEQ_SRC": "EXSEQ"}, lambda d: vex(d) & d["EXSTDT"].notna(), ["EXSTDT", "EXSEQ"], "first", manifest, "ADSL")
        adsl = derive_vars_merged(adsl, ex, ["STUDYID", "USUBJID"], {"TRTEDT": "EXENDT", "TRTEDTF": "EXENDTF", "TRTESEQ_SRC": "EXSEQ"}, lambda d: vex(d) & d["EXENDT"].notna(), ["EXENDT", "EXSEQ"], "last", manifest, "ADSL")
    else:
        adsl["TRTSDT"], adsl["TRTEDT"] = pd.NaT, pd.NaT

    if "RFSTDTC" in adsl.columns:
        adsl = derive_vars_dt(adsl, "RFSTDTC", "RFST", impute_dates)
        fb = adsl["TRTSDT"].isna() & adsl["RFSTDT"].notna()
        adsl.loc[fb, "TRTSDT"] = adsl.loc[fb, "RFSTDT"]
        if "RFENDTC" in adsl.columns:
            adsl = derive_vars_dt(adsl, "RFENDTC", "RFEN", impute_dates)
            adsl.loc[adsl["TRTEDT"].isna() & adsl["RFENDT"].notna(), "TRTEDT"] = adsl["RFENDT"]

    ts, te = pd.to_datetime(adsl["TRTSDT"], errors="coerce"), pd.to_datetime(adsl["TRTEDT"], errors="coerce")
    adsl["TRTDURD"] = np.where(ts.notna() & te.notna(), (te - ts).dt.days + 1, np.nan)

    ds_raw = bundle.get_domain("DS")
    if ds_raw is not None and not ds_raw.empty:
        ds, _ = combine_supp(ds_raw, bundle.get_domain("SUPPDS"), "DS")
        ds = derive_vars_dt(convert_blanks_to_na(ds), "DSSTDTC", "DSST", False)
        adsl = derive_vars_merged(
            adsl, ds, ["STUDYID", "USUBJID"], {"EOSDT": "DSSTDT", "DSDECOD_EOS": "DSDECOD", "DSTERM_EOS": "DSTERM", "EOSSEQ_SRC": "DSSEQ"},
            lambda d: (d["DSCAT"].astype(str).str.upper() == "DISPOSITION EVENT") & (d["DSDECOD"].astype(str).str.upper() != "SCREEN FAILURE"),
            ["DSSTDT", "DSSEQ"], "last", manifest, "ADSL",
        )
        adsl["EOSSTT"] = adsl["DSDECOD_EOS"].apply(lambda x: "ONGOING" if pd.isna(x) else ("COMPLETED" if str(x).upper() == "COMPLETED" else ("DISCONTINUED" if str(x).upper() != "SCREEN FAILURE" else np.nan)))
        adsl.loc[adsl["ARMCD"].astype(str).str.upper() == "SCRNFAIL", "EOSSTT"] = np.nan
        adsl["DCSREAS"] = np.where(adsl["EOSSTT"] == "DISCONTINUED", adsl["DSDECOD_EOS"], np.nan)
        adsl["DISCONFL"] = np.where(adsl["EOSSTT"] == "DISCONTINUED", "Y", np.nan)
        adsl = derive_vars_merged(adsl, ds, ["STUDYID", "USUBJID"], {"RANDDT": "DSSTDT"}, lambda d: d["DSDECOD"].astype(str).str.upper() == "RANDOMIZED", ["DSSTDT", "DSSEQ"], "first", manifest, "ADSL")
        adsl = adsl.drop(columns=["DSDECOD_EOS", "DSTERM_EOS"], errors="ignore")
    else:
        adsl["EOSDT"], adsl["EOSSTT"], adsl["DCSREAS"], adsl["DISCONFL"] = pd.NaT, "ONGOING", np.nan, np.nan

    adsl["AAGE"] = pd.to_numeric(adsl["AGE"], errors="coerce")
    adsl["AGEGR1"] = np.where(adsl["AAGE"].isna(), np.nan, np.where(adsl["AAGE"] < 65, "<65", np.where(adsl["AAGE"] <= 80, "65-80", ">80")))
    adsl["AGEGR1N"] = np.where(adsl["AAGE"].isna(), np.nan, np.where(adsl["AAGE"] < 65, 1, np.where(adsl["AAGE"] <= 80, 2, 3)))
    is_rand = ~adsl["ARMCD"].astype(str).str.upper().isin(["SCRNFAIL", "NOTASSGN", "NAN", ""])
    adsl["ITTFL"] = np.where(is_rand, "Y", "N")
    adsl["SAFFL"] = np.where((adsl["ITTFL"] == "Y") & adsl["TRTSDT"].notna(), "Y", "N")
    adsl["EFFFL"] = adsl.get("EFFFL", adsl["SAFFL"])
    adsl["COMP24FL"] = adsl.get("COMP24FL", np.where(adsl["EOSSTT"] == "COMPLETED", "Y", "N"))

    manifest.log("ADSL", ["TRT01P", "TRTSDT", "TRTEDT", "TRTDURD", "EOSSTT", "DCSREAS", "ITTFL", "SAFFL"], ["DM", "SUPPDM", "EX", "DS"], "Construct Subject-Level ADSL", "admiral::derive_vars_merged + derive_var_trtdurd", len(adsl))
    return adsl


def _finalize_bds(df: pd.DataFrame, seq_col: str, manifest: LineageManifest, ds_name: str) -> pd.DataFrame:
    out = df.copy()
    out["BASETYPE"] = "LAST"
    adt, ts, te = pd.to_datetime(out["ADT"], errors="coerce"), pd.to_datetime(out["TRTSDT"], errors="coerce"), pd.to_datetime(out["TRTEDT"], errors="coerce")
    out["ONTRTFL"] = np.where(adt.notna() & ts.notna() & (adt >= ts) & (te.isna() | (adt <= te)), "Y", np.nan)

    av, lo, hi = pd.to_numeric(out["AVAL"], errors="coerce"), pd.to_numeric(out.get("ANRLO"), errors="coerce"), pd.to_numeric(out.get("ANRHI"), errors="coerce")
    out["ANRIND"] = np.where(av.isna() | (lo.isna() & hi.isna()), np.nan, np.where(lo.notna() & (av < lo), "LOW", np.where(hi.notna() & (av > hi), "HIGH", "NORMAL")))

    out["ABLFL"] = np.nan
    bl_elig = out[av.notna() & ((adt.notna() & ts.notna() & (adt <= ts)) | (ts.isna() & out.get("VISIT", pd.Series("", index=out.index)).astype(str).str.upper().str.contains("BASELINE", na=False)))].copy()
    if not bl_elig.empty:
        ocols = [c for c in ["ADT", "ATPTN", "VISITNUM", seq_col] if c in bl_elig.columns]
        bl_idx = bl_elig.sort_values(by=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD"] + ocols).drop_duplicates(subset=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD"], keep="last").index
        out.loc[bl_idx, "ABLFL"] = "Y"

    b_rows = out[out["ABLFL"] == "Y"][["STUDYID", "USUBJID", "BASETYPE", "PARAMCD", "AVAL", "ANRIND"]].drop_duplicates(subset=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD"], keep="last").rename(columns={"AVAL": "BASE", "ANRIND": "BNRIND"})
    out = out.merge(b_rows, on=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD"], how="left")

    base_n = pd.to_numeric(out["BASE"], errors="coerce")
    out["CHG"] = av - base_n
    zmask = av.notna() & base_n.notna() & (base_n == 0)
    if zmask.any():
        manifest.zero_denominators.append({"dataset": ds_name, "variable": "PCHG", "count": int(zmask.sum())})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["PCHG"] = np.where(av.notna() & base_n.notna() & (base_n != 0), ((av - base_n) / np.abs(base_n)) * 100.0, np.nan)
        out["R2BASE"] = np.where(av.notna() & base_n.notna() & (base_n != 0), av / base_n, np.nan)
    out["SHIFT1"] = np.where(out["BNRIND"].notna() & out["ANRIND"].notna(), out["BNRIND"].astype(str) + " to " + out["ANRIND"].astype(str), np.nan)

    out["ANL01FL"] = np.nan
    anl_elig = out[av.notna() & out["AVISIT"].notna() & ~out["AVISIT"].astype(str).str.upper().str.contains("UNSCHED", na=False)].copy()
    if not anl_elig.empty:
        ocols = [c for c in ["ADT", "ATPTN", "VISITNUM", seq_col] if c in anl_elig.columns]
        a_idx = anl_elig.sort_values(by=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD", "AVISIT"] + ocols).drop_duplicates(subset=["STUDYID", "USUBJID", "BASETYPE", "PARAMCD", "AVISIT"], keep="last").index
        out.loc[a_idx, "ANL01FL"] = "Y"

    s_cols = [c for c in ["PARAMN", "PARAMCD", "ADT", "AVISITN", "ATPTN", seq_col] if c in out.columns]
    out = out.sort_values(by=["STUDYID", "USUBJID"] + s_cols, na_position="last").reset_index(drop=True)
    out["ASEQ"] = out.groupby(["STUDYID", "USUBJID"]).cumcount() + 1
    manifest.log(ds_name, ["ABLFL", "BASE", "CHG", "PCHG", "ANRIND", "SHIFT1", "ANL01FL", "ASEQ"], [ds_name[2:], "ADSL"], f"Derive {ds_name} BDS Variables", "admiral::derive_var_extreme_flag + derive_var_base + derive_var_chg", len(out))
    return out


def build_bds_datasets(
    bundle: IngestionBundle, adsl: pd.DataFrame, requested: List[str], manifest: LineageManifest
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    outputs: Dict[str, pd.DataFrame] = {"ADSL": adsl}
    status: Dict[str, str] = {"ADSL": f"SUCCESS ({len(adsl)} rows)"}
    adsl_cols = [c for c in ["STUDYID", "USUBJID", "SUBJID", "SITEID", "AGE", "AGEGR1", "SEX", "RACE", "TRT01P", "TRT01PN", "TRT01A", "TRTSDT", "TRTEDT", "EOSSTT", "EOSDT", "SAFFL", "ITTFL", "STUDYEYE"] if c in adsl.columns]

    if "ADVS" in requested:
        vs = bundle.get_domain("VS")
        if vs is None or vs.empty:
            status["ADVS"] = "UNAVAILABLE: Source domain VS is missing."
        else:
            advs = convert_blanks_to_na(vs).drop(columns=[c for c in adsl_cols if c not in ("STUDYID", "USUBJID")], errors="ignore").merge(adsl[adsl_cols], on=["STUDYID", "USUBJID"], how="left")
            advs = derive_vars_dy(derive_vars_dt(advs, "VSDTC", "A"), "TRTSDT", "ADT", "ADY")
            advs["PARAMCD"] = advs["VSTESTCD"].astype(str).str.upper()
            advs["AVAL"] = pd.to_numeric(advs["VSSTRESN"], errors="coerce")
            advs["AVALC"] = advs["VSSTRESC"].astype(str)
            advs["DTYPE"], advs["SRCDOM"], advs["SRCVAR"], advs["SRCSEQ"] = np.nan, "VS", "VSSTRESN", advs["VSSEQ"]
            advs["AVISIT"] = np.where(advs["VISIT"].isna() | advs["VISIT"].astype(str).str.upper().str.contains("SCREEN|UNSCHED", na=False), np.nan, advs["VISIT"].astype(str).str.title())
            advs["AVISITN"] = pd.to_numeric(advs.get("VISITNUM"), errors="coerce")
            advs["ATPTN"] = pd.to_numeric(advs.get("VSTPTNUM"), errors="coerce")

            bp = advs[advs["PARAMCD"].isin(["SYSBP", "DIABP"]) & advs["AVAL"].notna()]
            if not bp.empty:
                gcols = [c for c in ["STUDYID", "USUBJID", "VISIT", "VISITNUM", "ADT", "ADY", "AVISIT", "AVISITN", "ATPTN"] if c in bp.columns]
                piv = bp.pivot_table(index=gcols, columns="PARAMCD", values="AVAL", aggfunc="last").reset_index().dropna(subset=["SYSBP", "DIABP"])
                if not piv.empty:
                    piv["AVAL"] = (piv["SYSBP"] + 2.0 * piv["DIABP"]) / 3.0
                    piv["AVALC"], piv["PARAMCD"], piv["VSTESTCD"], piv["DTYPE"], piv["SRCDOM"], piv["SRCVAR"], piv["SRCSEQ"] = piv["AVAL"].round(2).astype(str), "MAP", "MAP", "CALCULATION", "VS", "SYSBP+DIABP", np.nan
                    piv = piv.drop(columns=["SYSBP", "DIABP"]).merge(adsl[adsl_cols], on=["STUDYID", "USUBJID"], how="left")
                    advs = pd.concat([advs, piv], ignore_index=True)

            advs = advs.merge(ADVS_LOOKUP, on="PARAMCD", how="left")
            advs = _finalize_bds(advs, "VSSEQ", manifest, "ADVS")
            outputs["ADVS"] = advs
            status["ADVS"] = f"SUCCESS ({len(advs)} rows)"

    if "ADLB" in requested:
        lb = bundle.get_domain("LB")
        if lb is None or lb.empty:
            status["ADLB"] = "UNAVAILABLE: Source domain LB is missing."
        else:
            lb_c, _ = combine_supp(lb, bundle.get_domain("SUPPLB"), "LB")
            adlb = convert_blanks_to_na(lb_c).drop(columns=[c for c in adsl_cols if c not in ("STUDYID", "USUBJID")], errors="ignore").merge(adsl[adsl_cols], on=["STUDYID", "USUBJID"], how="left")
            adlb = derive_vars_dy(derive_vars_dt(adlb, "LBDTC", "A"), "TRTSDT", "ADT", "ADY")
            adlb["PARAMCD"] = adlb["LBTESTCD"].astype(str).str.upper()
            adlb["PARAM"] = adlb["LBTEST"].astype(str) + " (" + adlb["LBSTRESU"].fillna("").astype(str) + ")"
            adlb["PARAMN"] = adlb["PARAMCD"].astype("category").cat.codes + 1
            adlb["PARCAT1"] = adlb.get("LBCAT", "LABORATORY")
            adlb["AVAL"] = pd.to_numeric(adlb["LBSTRESN"], errors="coerce")
            adlb["AVALC"] = adlb["LBSTRESC"].astype(str)
            adlb["ANRLO"], adlb["ANRHI"] = pd.to_numeric(adlb.get("LBSTNRLO"), errors="coerce"), pd.to_numeric(adlb.get("LBSTNRHI"), errors="coerce")
            adlb["DTYPE"], adlb["SRCDOM"], adlb["SRCVAR"], adlb["SRCSEQ"] = np.nan, "LB", "LBSTRESN", adlb["LBSEQ"]
            adlb["AVISIT"] = np.where(adlb["VISIT"].isna() | adlb["VISIT"].astype(str).str.upper().str.contains("UNSCHED", na=False), np.nan, adlb["VISIT"].astype(str).str.title())
            adlb["AVISITN"], adlb["ATPTN"] = pd.to_numeric(adlb.get("VISITNUM"), errors="coerce"), 1
            adlb = _finalize_bds(adlb, "LBSEQ", manifest, "ADLB")
            outputs["ADLB"] = adlb
            status["ADLB"] = f"SUCCESS ({len(adlb)} rows)"

    if "ADEG" in requested:
        eg = bundle.get_domain("EG")
        if eg is None or eg.empty:
            status["ADEG"] = "UNAVAILABLE: Source SDTM domain 'EG' is not present in this study (e.g., PHUSE Pilot 01)."
        else:
            adeg = convert_blanks_to_na(eg).merge(adsl[adsl_cols], on=["STUDYID", "USUBJID"], how="left")
            adeg = derive_vars_dy(derive_vars_dt(adeg, "EGDTC", "A"), "TRTSDT", "ADT", "ADY")
            adeg["PARAMCD"] = adeg["EGTESTCD"].astype(str).str.upper()
            adeg["PARAM"] = adeg.get("EGTEST", adeg["PARAMCD"]).astype(str)
            adeg["PARAMN"] = adeg["PARAMCD"].astype("category").cat.codes + 1
            adeg["AVAL"] = pd.to_numeric(adeg["EGSTRESN"], errors="coerce")
            adeg["AVISIT"], adeg["AVISITN"], adeg["ATPTN"] = adeg.get("VISIT", "Visit 1"), pd.to_numeric(adeg.get("VISITNUM", 1), errors="coerce"), 1
            adeg["ANRLO"], adeg["ANRHI"], adeg["DTYPE"] = np.nan, np.nan, np.nan
            adeg["SRCDOM"], adeg["SRCVAR"], adeg["SRCSEQ"] = "EG", "EGSTRESN", adeg.get("EGSEQ", 1)
            adeg = _finalize_bds(adeg, "EGSEQ", manifest, "ADEG")
            outputs["ADEG"] = adeg
            status["ADEG"] = f"SUCCESS ({len(adeg)} rows)"

    if "ADBCVA" in requested:
        oe = bundle.get_domain("OE")
        if oe is None or oe.empty:
            status["ADBCVA"] = "UNAVAILABLE: Source SDTM domain 'OE' (Ophthalmology) is not present in this study."
        else:
            oe_c = convert_blanks_to_na(oe)
            if "OETESTCD" in oe_c.columns:
                oe_c = oe_c[oe_c["OETESTCD"].astype(str).str.upper() == "VACSCORE"].copy()
            adsl_oe = adsl.copy()
            if "STUDYEYE" not in adsl_oe.columns:
                adsl_oe["STUDYEYE"] = "RIGHT"
            adbcva = oe_c.merge(adsl_oe[adsl_cols + (["STUDYEYE"] if "STUDYEYE" not in adsl_cols else [])], on=["STUDYID", "USUBJID"], how="left")
            adbcva = derive_vars_dy(derive_vars_dt(adbcva, "OEDTC", "A"), "TRTSDT", "ADT", "ADY")
            lat = adbcva.get("OELAT", pd.Series("RIGHT", index=adbcva.index)).astype(str).str.upper()
            adbcva["AFEYE"] = np.where(lat == adbcva["STUDYEYE"].astype(str).str.upper(), "Study Eye", "Fellow Eye")
            adbcva["PARAMCD"] = np.where(adbcva["AFEYE"] == "Study Eye", "SBCVA", "FBCVA")
            adbcva["PARAM"] = np.where(adbcva["PARAMCD"] == "SBCVA", "Study Eye Visual Acuity Score (letters)", "Fellow Eye Visual Acuity Score (letters)")
            adbcva["PARAMN"] = np.where(adbcva["PARAMCD"] == "SBCVA", 1, 2)
            adbcva["AVAL"] = pd.to_numeric(adbcva["OESTRESN"], errors="coerce")
            adbcva["DTYPE"], adbcva["AVISIT"], adbcva["AVISITN"], adbcva["ATPTN"] = np.nan, adbcva.get("VISIT", "Visit 1"), pd.to_numeric(adbcva.get("VISITNUM", 1), errors="coerce"), 1
            adbcva["ANRLO"], adbcva["ANRHI"], adbcva["SRCDOM"], adbcva["SRCVAR"], adbcva["SRCSEQ"] = 70.0, 100.0, "OE", "OESTRESN", adbcva.get("OESEQ", 1)

            log_df = adbcva[adbcva["AVAL"].notna()].copy()
            log_df["AVAL"] = (-0.02 * log_df["AVAL"] + 1.7).round(4)
            log_df["PARAMCD"] = np.where(log_df["PARAMCD"] == "SBCVA", "SBCVALOG", "FBCVALOG")
            log_df["PARAM"] = np.where(log_df["PARAMCD"] == "SBCVALOG", "Study Eye Visual Acuity LogMAR Score", "Fellow Eye Visual Acuity LogMAR Score")
            log_df["PARAMN"] = np.where(log_df["PARAMCD"] == "SBCVALOG", 3, 4)
            log_df["DTYPE"] = "LOGMAR"
            for sc in ("VISIT", "VISITNUM", "OEDY", "OEDTC"):
                if sc in log_df.columns:
                    log_df[sc] = np.nan
            adbcva = pd.concat([adbcva, log_df], ignore_index=True)
            adbcva = _finalize_bds(adbcva, "OESEQ", manifest, "ADBCVA")
            adbcva["CRIT1"] = "Gain of >= 5 and <= 10 letters relative to baseline"
            adbcva["CRIT1FL"] = np.where(adbcva["PARAMCD"].isin(["SBCVA", "FBCVA"]) & adbcva["CHG"].between(5, 10), "Y", "N")
            outputs["ADBCVA"] = adbcva
            status["ADBCVA"] = f"SUCCESS ({len(adbcva)} rows)"

    return outputs, status
