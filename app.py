from pathlib import Path
import streamlit as st
from clinical_aggregator.agent import AgentRequest, ClinicalStudyAgent
from clinical_aggregator.ingestion import PHUSE_PILOT01_URL

st.set_page_config(page_title="Clinical Study Source Data Aggregator Agent", layout="wide")
st.title("Clinical Study Source Data Aggregator - AI Agent")
st.caption("CDISC SDTM-to-ADaM Transformation ({admiral}), Validation (define.xml), Dropout Survival & Leakage Prevention")

with st.sidebar:
    st.header("1. Required Agent Inputs")
    source_link = st.text_input("SDTM Source Link / Path", value=PHUSE_PILOT01_URL)
    target_tables = st.multiselect(
        "Required ADaM Tables to Aggregate",
        options=["ADSL", "ADVS", "ADLB", "ADEG", "ADBCVA"],
        default=["ADSL", "ADVS", "ADLB", "ADEG", "ADBCVA"],
    )
    output_loc = st.text_input("Output Folder Location", value="./agent_outputs")
    approve_imp = st.checkbox("Approve Month-Level Partial Date Imputation", value=False)
    run_btn = st.button("Run AI Agent & Generate ADaM Tables", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Or Prompt the Agent Directly")
    nl_prompt = st.text_area(
        "Natural Language Agent Prompt",
        value=f"Read SDTM data from {PHUSE_PILOT01_URL}, aggregate ADSL, ADVS, and ADLB, and save outputs to ./agent_outputs",
        height=110,
    )
    run_nl_btn = st.button("Execute Natural Language Prompt", use_container_width=True)

agent = ClinicalStudyAgent()

if run_nl_btn:
    req, missing = agent.parse_natural_language_prompt(nl_prompt)
    if missing:
        st.error("Please provide all 3 required parameters in your prompt:\n" + "\n".join(f"- {m}" for m in missing))
    else:
        with st.spinner("Executing AI Agent Pipeline..."):
            st.session_state["agent_result"] = agent.execute_request(req)

if run_btn:
    if not source_link.strip() or not target_tables or not output_loc.strip():
        st.error("All 3 inputs (SDTM Source Link, Target ADaM Tables, and Output Folder Location) are required.")
    else:
        req = AgentRequest(
            source_sdtm_link=source_link.strip(),
            target_adam_tables=target_tables,
            output_location=output_loc.strip(),
            approve_partial_date_imputation=approve_imp,
        )
        with st.spinner("Ingesting SDTM, validating define.xml, building ADaM tables, and writing to output folder..."):
            st.session_state["agent_result"] = agent.execute_request(req)

if "agent_result" in st.session_state:
    res = st.session_state["agent_result"]
    st.success(f"All requested ADaM tables and reports have been written to: `{res['output_location']}`")

    with open(res["zip_bundle_path"], "rb") as f:
        st.download_button(
            "Download Complete Output Bundle (.zip)",
            data=f.read(),
            file_name=Path(res["zip_bundle_path"]).name,
            mime="application/zip",
        )

    tabs = st.tabs([
        "1. ADaM Tables & Status",
        "2. SDTM Validation",
        "3. Lineage & Manifest",
        "4. Missingness Report",
        "5. Dropout, Survival & Leakage Guard",
        "6. Gemini 4 Argon Summary",
    ])

    with tabs[0]:
        st.subheader("ADaM Transformation Status")
        st.json(res["adam_status"])
        for ds_name, df in res["adam_datasets"].items():
            st.markdown(f"#### `{ds_name}` ({len(df):,} rows x {len(df.columns)} columns)")
            st.dataframe(df.head(50), use_container_width=True)

    with tabs[1]:
        vrep = res["validation_report"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical", vrep.critical_count)
        c2.metric("Errors", vrep.error_count)
        c3.metric("Warnings", vrep.warning_count)
        c4.metric("Define.xml Readable", str(vrep.define_readable))
        st.dataframe(vrep.to_dataframe(), use_container_width=True)

    with tabs[2]:
        st.subheader("Deterministic R {admiral}-to-Python Derivation Lineage")
        st.dataframe(res["lineage_manifest"], use_container_width=True)

    with tabs[3]:
        st.subheader("Expected vs. Observed vs. Post-Discontinuation Missingness")
        st.dataframe(res["missingness_df"], use_container_width=True)

    with tabs[4]:
        dr = res["dropout_results"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Discontinuation Rates by Planned Arm")
            st.dataframe(dr["by_arm"], use_container_width=True)
            st.markdown("#### Cox Proportional Hazards")
            st.dataframe(dr["cox_df"], use_container_width=True)
        with col_b:
            if Path(dr["km_plot_path"]).exists():
                st.image(dr["km_plot_path"])
        st.markdown("#### FR-06 Data Leakage Prevention Audit & Baseline ML Holdout Metrics")
        st.json(dr["leakage_audit"])

    with tabs[5]:
        st.markdown(res["ai_narrative"])
