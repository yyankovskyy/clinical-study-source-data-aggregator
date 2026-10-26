from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re
import xml.etree.ElementTree as ET
import pandas as pd
from clinical_aggregator.ingestion import IngestionBundle

ISO_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?)?)?$")
REQUIRED_DOMAINS: Dict[str, List[str]] = {
    "DM": ["STUDYID", "USUBJID", "SUBJID", "SITEID", "AGE", "SEX", "RACE", "ARMCD", "ARM"],
    "DS": ["STUDYID", "USUBJID", "DSSEQ", "DSTERM", "DSDECOD", "DSCAT", "DSSTDTC"],
    "EX": ["STUDYID", "USUBJID", "EXSEQ", "EXTRT", "EXDOSE", "EXSTDTC", "EXENDTC"],
    "VS": ["STUDYID", "USUBJID", "VSSEQ", "VSTESTCD", "VSTEST", "VSSTRESN", "VSSTRESU", "VSDTC"],
    "LB": ["STUDYID", "USUBJID", "LBSEQ", "LBTESTCD", "LBTEST", "LBSTRESN", "LBSTRESU", "LBDTC"],
}
DOMAIN_KEYS: Dict[str, List[str]] = {
    "DM": ["STUDYID", "USUBJID"], "DS": ["STUDYID", "USUBJID", "DSSEQ"],
    "EX": ["STUDYID", "USUBJID", "EXSEQ"], "VS": ["STUDYID", "USUBJID", "VSSEQ"],
    "LB": ["STUDYID", "USUBJID", "LBSEQ"], "QS": ["STUDYID", "USUBJID", "QSSEQ"],
}


@dataclass
class ValidationFinding:
    check_id: str
    severity: str
    domain: str
    variable: str
    message: str
    affected_records: int
    evidence: str = ""


@dataclass
class ValidationReport:
    study_id: str
    define_readable: bool
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    can_proceed: bool = True
    findings: List[ValidationFinding] = field(default_factory=list)

    def add(self, cid: str, sev: str, dom: str, var: str, msg: str, cnt: int, ev: str = "") -> None:
        s = sev.upper()
        if s == "CRITICAL":
            self.critical_count += 1
            self.can_proceed = False
        elif s == "ERROR":
            self.error_count += 1
        elif s == "WARNING":
            self.warning_count += 1
        else:
            self.info_count += 1
        self.findings.append(ValidationFinding(cid, s, dom, var, msg, int(cnt), ev))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.findings]) if self.findings else pd.DataFrame()


def parse_define_xml(define_path: Optional[Path]) -> Tuple[bool, str, Dict[str, Set[str]], Dict[str, Dict[str, str]]]:
    if define_path is None or not Path(define_path).exists():
        return False, "Missing define.xml", {}, {}
    try:
        root = ET.parse(define_path).getroot()
    except Exception as exc:
        return False, str(exc), {}, {}
    strip = lambda t: t.split("}", 1)[1] if "}" in t else t
    study_name = "CDISCPILOT01"
    codelists: Dict[str, Set[str]] = {}
    item_cl: Dict[str, str] = {}
    item_name: Dict[str, str] = {}
    dom_var_cl: Dict[str, Dict[str, str]] = {}

    for el in root.iter():
        tag = strip(el.tag)
        if tag == "StudyName" and el.text:
            study_name = el.text.strip()
        elif tag == "CodeList":
            oid = el.attrib.get("OID", "")
            vals = {c.attrib["CodedValue"].strip() for c in el if strip(c.tag) in ("CodeListItem", "EnumeratedItem") and "CodedValue" in c.attrib}
            if oid:
                codelists[oid] = vals
        elif tag == "ItemDef":
            oid, name = el.attrib.get("OID", ""), el.attrib.get("Name", "").upper()
            item_name[oid] = name
            for c in el:
                if strip(c.tag) == "CodeListRef":
                    item_cl[oid] = c.attrib.get("CodeListOID", "")
    for el in root.iter():
        if strip(el.tag) == "ItemGroupDef":
            dname = el.attrib.get("Name", "").upper()
            dom_var_cl[dname] = {}
            for c in el:
                if strip(c.tag) == "ItemRef":
                    ioid = c.attrib.get("ItemOID", "")
                    if ioid in item_cl and ioid in item_name:
                        dom_var_cl[dname][item_name[ioid]] = item_cl[ioid]
    return True, study_name, codelists, dom_var_cl


def validate_sdtm(bundle: IngestionBundle) -> ValidationReport:
    ok_xml, study_name, codelists, dom_var_cl = parse_define_xml(bundle.define_xml_path)
    dm = bundle.domains.get("DM")
    sid = str(dm["STUDYID"].dropna().iloc[0]) if dm is not None and "STUDYID" in dm.columns and not dm.empty else study_name
    rep = ValidationReport(study_id=sid, define_readable=ok_xml)

    if ok_xml:
        rep.add("DEF-01", "INFO", "DEFINE.XML", "ALL", f"Parsed define.xml ({study_name}, {len(codelists)} CodeLists).", len(codelists))
    else:
        rep.add("DEF-01", "WARNING", "DEFINE.XML", "ALL", f"define.xml issue: {study_name}", 0)

    if dm is None or dm.empty:
        rep.add("DOM-01", "CRITICAL", "DM", "ALL", "Mandatory DM domain is missing or empty.", 0)
        return rep

    for dom, req_cols in REQUIRED_DOMAINS.items():
        if dom not in bundle.domains:
            rep.add(f"DOM-{dom}", "CRITICAL" if dom in ("DM", "DS", "EX") else "WARNING", dom, "ALL", f"Missing domain {dom}", 0)
            continue
        df = bundle.domains[dom]
        miss = [c for c in req_cols if c not in df.columns]
        if miss:
            rep.add(f"VAR-{dom}", "CRITICAL" if "USUBJID" in miss else "ERROR", dom, ",".join(miss), f"Missing required columns in {dom}", len(df))

    dm_subjs = set(dm["USUBJID"].dropna().astype(str))
    dup_dm = dm[dm.duplicated(subset=["USUBJID"], keep=False)]
    if not dup_dm.empty:
        rep.add("KEY-DM", "CRITICAL", "DM", "USUBJID", "Duplicate USUBJID in DM.", len(dup_dm))

    for dom, df in bundle.domains.items():
        if "USUBJID" in df.columns and dom != "DM":
            orph = set(df["USUBJID"].dropna().astype(str)) - dm_subjs
            if orph:
                rep.add(f"REF-{dom}", "ERROR", dom, "USUBJID", f"{len(orph)} USUBJIDs in {dom} not in DM.", int(df["USUBJID"].isin(orph).sum()), str(sorted(list(orph))[:5]))

        kcols = DOMAIN_KEYS.get(dom)
        if kcols and all(k in df.columns for k in kcols):
            dups = df[df.duplicated(subset=kcols, keep=False)]
            if not dups.empty:
                rep.add(f"KEY-{dom}", "ERROR", dom, "+".join(kcols), f"Duplicate key in {dom}", len(dups))

        for dcol in [c for c in df.columns if c.endswith("DTC")]:
            s = df[dcol].dropna().astype(str).str.strip()
            s = s[s != ""]
            bad = s[~s.str.match(ISO_RE)]
            if not bad.empty:
                rep.add(f"DAT-{dom}-{dcol}", "ERROR", dom, dcol, "Invalid ISO-8601 date format.", len(bad), str(bad.unique()[:3].tolist()))
            part = s[s.str.len() < 10]
            if not part.empty:
                rep.add(f"PRT-{dom}-{dcol}", "WARNING", dom, dcol, "Partial dates detected (not imputed without approval).", len(part))

        if ok_xml and dom in dom_var_cl:
            for col, cl_oid in dom_var_cl[dom].items():
                if col in df.columns and cl_oid in codelists:
                    s = df[col].dropna().astype(str).str.strip()
                    s = s[s != ""]
                    outside = s[~s.isin(codelists[cl_oid])]
                    if not outside.empty:
                        rep.add(f"CT-{dom}-{col}", "WARNING", dom, col, f"Values outside CodeList {cl_oid}", len(outside), str(outside.unique()[:4].tolist()))

        tcol, ucol = f"{dom}TESTCD", f"{dom}STRESU"
        if tcol in df.columns and ucol in df.columns:
            vu = df.dropna(subset=[tcol, ucol]).copy()
            vu[ucol] = vu[ucol].astype(str).str.strip()
            vu = vu[vu[ucol] != ""]
            multi = vu.groupby(tcol)[ucol].nunique()
            m_tests = multi[multi > 1].index.tolist()
            if m_tests:
                rep.add(f"UNT-{dom}", "ERROR", dom, ucol, f"Multiple units for {m_tests}", int(df[tcol].isin(m_tests).sum()))

    return rep
