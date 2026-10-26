import hashlib
import io
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

PHUSE_PILOT01_URL = "https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01"
PHUSE_FILES = [
    "define.xml", "dm.xpt", "ds.xpt", "ex.xpt", "vs.xpt", "lb.xpt", "qs.xpt",
    "ae.xpt", "cm.xpt", "mh.xpt", "sc.xpt", "se.xpt", "sv.xpt", "ta.xpt",
    "te.xpt", "ti.xpt", "ts.xpt", "tv.xpt", "relrec.xpt", "suppdm.xpt",
    "suppds.xpt", "supplb.xpt", "suppae.xpt",
]


@dataclass
class IngestionBundle:
    source_type: str
    source_location: str
    working_dir: Path
    domains: Dict[str, pd.DataFrame] = field(default_factory=dict)
    supp_domains: Dict[str, pd.DataFrame] = field(default_factory=dict)
    define_xml_path: Optional[Path] = None
    file_hashes_sha256: Dict[str, str] = field(default_factory=dict)
    inventory_summary: List[Dict[str, object]] = field(default_factory=list)

    def get_domain(self, name: str) -> Optional[pd.DataFrame]:
        k = name.upper()
        if k in self.domains:
            return self.domains[k].copy(deep=True)
        if k in self.supp_domains:
            return self.supp_domains[k].copy(deep=True)
        return None


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_immutability(directory: Path, hashes: Dict[str, str]) -> Tuple[bool, List[str]]:
    bad = []
    for rel, exp in hashes.items():
        p = directory / rel
        if not p.exists() or compute_sha256(p) != exp:
            bad.append(rel)
    return (len(bad) == 0, bad)


def safe_extract_zip(zip_src: Union[Path, io.BytesIO], target_dir: Path) -> Path:
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_src, "r") as zf:
        for m in zf.infolist():
            if m.filename.startswith(("/", "\\")) or ".." in Path(m.filename).parts:
                raise ValueError(f"Unsafe ZIP path blocked: {m.filename}")
            dest = (target_dir / m.filename).resolve()
            if not str(dest).startswith(str(target_dir)):
                raise ValueError(f"ZIP entry escaped target directory: {m.filename}")
            if m.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m, "r") as s, open(dest, "wb") as d:
                    shutil.copyfileobj(s, d)
    return target_dir


def read_sdtm_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".xpt":
        df = pd.read_sas(path, format="xport")
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].apply(
                    lambda x: x.decode("utf-8", errors="replace").strip()
                    if isinstance(x, (bytes, bytearray))
                    else (x.strip() if isinstance(x, str) else x)
                )
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    df.columns = [str(c).upper().strip() for c in df.columns]
    return df


def download_github_sdtm(github_url: str, cache_dir: Optional[Path] = None) -> Path:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)", github_url.strip().rstrip("/"))
    if not m:
        raise ValueError(f"Invalid GitHub tree URL: {github_url}")
    owner, repo, branch, folder = m.groups()
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "clinical_agent_sdtm_cache" / f"{owner}_{repo}_{Path(folder).name}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if len(list(cache_dir.glob("*.xpt")) + list(cache_dir.glob("*.csv"))) >= 6 and (cache_dir / "define.xml").exists():
        return cache_dir

    files: List[Tuple[str, str]] = []
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}?ref={branch}"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "Clinical-SDTM-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            for item in json.loads(resp.read().decode("utf-8")):
                n, u = item.get("name", ""), item.get("download_url")
                if u and n.lower().endswith((".xpt", ".csv", ".xml")):
                    files.append((n, u))
    except Exception:
        base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{folder}"
        files = [(f, f"{base}/{f}") for f in PHUSE_FILES]

    for fname, url in files:
        dest = cache_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Clinical-SDTM-Agent/2.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as out:
                shutil.copyfileobj(r, out)
        except Exception:
            dest.unlink(missing_ok=True)
    return cache_dir


def ingest_sdtm_source(source: str, cache_dir: Optional[Path] = None) -> IngestionBundle:
    src = source.strip()
    if src.startswith(("http://", "https://")):
        work_dir = download_github_sdtm(src, cache_dir=cache_dir)
        stype = "github"
    else:
        p = Path(src).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".zip":
            work_dir = safe_extract_zip(p, Path(tempfile.mkdtemp(prefix="sdtm_zip_")))
            stype = "zip"
        elif p.is_dir():
            work_dir = p
            stype = "local_dir"
        else:
            raise FileNotFoundError(f"SDTM source not found or unsupported: {source}")

    bundle = IngestionBundle(source_type=stype, source_location=src, working_dir=work_dir)
    xmls = list(work_dir.rglob("define*.xml"))
    if xmls:
        bundle.define_xml_path = xmls[0]
        bundle.file_hashes_sha256[str(xmls[0].relative_to(work_dir))] = compute_sha256(xmls[0])

    data_files = sorted(
        list(work_dir.rglob("*.xpt")) + list(work_dir.rglob("*.XPT"))
        + list(work_dir.rglob("*.csv")) + list(work_dir.rglob("*.CSV"))
    )
    seen = set()
    for fpath in data_files:
        if fpath.resolve() in seen:
            continue
        seen.add(fpath.resolve())
        rel = str(fpath.relative_to(work_dir))
        h = compute_sha256(fpath)
        bundle.file_hashes_sha256[rel] = h
        dom = fpath.stem.upper()
        df = read_sdtm_file(fpath)
        if dom.startswith("SUPP") or dom == "RELREC":
            bundle.supp_domains[dom] = df
        else:
            bundle.domains[dom] = df
        bundle.inventory_summary.append({
            "Domain": dom, "File": fpath.name, "Rows": len(df), "Columns": len(df.columns),
            "Unique_Subjects": int(df["USUBJID"].nunique()) if "USUBJID" in df.columns else 0,
            "SHA256_Prefix": h[:12],
        })
    return bundle


def combine_supp(parent_df: pd.DataFrame, supp_df: Optional[pd.DataFrame], domain: str) -> Tuple[pd.DataFrame, List[str]]:
    if supp_df is None or supp_df.empty:
        return parent_df.copy(), []
    df, supp = parent_df.copy(), supp_df.copy()
    if "RDOMAIN" in supp.columns:
        supp = supp[supp["RDOMAIN"].astype(str).str.upper() == domain.upper()]
    if supp.empty or "QNAM" not in supp.columns or "QVAL" not in supp.columns:
        return df, []
    added: List[str] = []
    idvar = supp["IDVAR"].fillna("").astype(str).str.strip() if "IDVAR" in supp.columns else pd.Series([""] * len(supp), index=supp.index)
    if (idvar == "").any():
        piv = supp[idvar == ""].pivot_table(index=["STUDYID", "USUBJID"], columns="QNAM", values="QVAL", aggfunc="first").reset_index()
        qcols = [c for c in piv.columns if c not in ("STUDYID", "USUBJID") and c not in df.columns]
        added.extend(qcols)
        df = df.merge(piv[["STUDYID", "USUBJID"] + qcols], on=["STUDYID", "USUBJID"], how="left")
    if (idvar != "").any() and "IDVARVAL" in supp.columns:
        for idv, grp in supp[idvar != ""].groupby("IDVAR"):
            col = str(idv).strip().upper()
            if col in df.columns:
                g = grp.copy()
                if pd.api.types.is_numeric_dtype(df[col]):
                    g[col] = pd.to_numeric(g["IDVARVAL"], errors="coerce")
                else:
                    g[col] = g["IDVARVAL"].astype(str)
                piv = g.pivot_table(index=["STUDYID", "USUBJID", col], columns="QNAM", values="QVAL", aggfunc="first").reset_index()
                qcols = [c for c in piv.columns if c not in ("STUDYID", "USUBJID", col) and c not in df.columns]
                added.extend(qcols)
                df = df.merge(piv[["STUDYID", "USUBJID", col] + qcols], on=["STUDYID", "USUBJID", col], how="left")
    return df, added
