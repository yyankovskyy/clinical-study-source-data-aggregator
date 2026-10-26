# Clinical Study Source Data Aggregator (AI Agent & SDTM-to-ADaM Pipeline)

An AI-orchestrated, deterministic clinical data engineering and biostatistics pipeline that transforms **CDISC SDTM** study data (such as **PHUSE Pilot 01 / `cdiscpilot01`**) into analysis-ready **ADaM** datasets (`ADSL`, `ADVS`, `ADLB`, `ADEG`, `ADBCVA`), validates `define.xml` compliance, performs dropout-aware missingness and survival analysis, and enforces strict data-leakage prevention.

## Key Features
- **3-Parameter AI Agent Interface (`run_agent.py` & `app.py`)**: Prompts the user for:
  1. **Source SDTM Link/Path** (GitHub folder URL, `.zip` archive, or local directory)
  2. **Required ADaM Tables** (`ADSL`, `ADVS`, `ADLB`, `ADEG`, `ADBCVA`)
  3. **Output Location** (local folder where all generated tables, plots, and reports are saved)
- **Deterministic Pharmaverse `{admiral}` & `{admiralophtha}` Derivations**: No Study Day 0, verified join cardinality, baseline flags (`ABLFL`), `BASE`, `CHG`, `PCHG` (with zero-denominator guard), `ANRIND`, `SHIFT1`, `ANL01FL`, `ASEQ`, and LogMAR derivations.
- **Dropout & Survival Analysis**: Kaplan-Meier retention curves, Cox Proportional Hazards, Competing-Risks Cumulative Incidence (`CIF`), and `LeakageGuard`-audited baseline ML prediction.

## Quick Start

    pip install -r requirements.txt
    cp .env.example .env

    # 1. Run Automated Verification Tests
    pytest -v

    # 2. Run the CLI AI Agent against PHUSE Pilot 01
    python3 run_agent.py \
      --source "https://github.com/phuse-org/phuse-scripts/tree/master/data/sdtm/cdiscpilot01" \
      --tables ADSL ADVS ADLB ADEG ADBCVA \
      --output-dir "./my_study_outputs"

    # 3. Launch the Interactive Streamlit App
    streamlit run app.py
