import streamlit as st
import tempfile
import json

from rag import (
    load_document,
    chunk_text,
    build_faiss_index
)

from compliance import (
    generate_regulation_inventory,
    run_compliance_review,
    export_to_excel
)

# ---------------------------------

st.set_page_config(
    page_title="Regulatory Reviewer",
    layout="wide"
)

st.title(
    "📋 Regulatory Compliance Reviewer"
)

# ---------------------------------
# REGULATION FILE
# ---------------------------------

reg_file = st.file_uploader(
    "Upload Regulatory File",
    type=["pdf"]
)

if reg_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(reg_file.read())

        regulation_path = tmp.name

    regulatory_text = load_document(
        regulation_path
    )

    if st.button(
        "Generate Requirements"
    ):

        with st.spinner(
            "Extracting requirements..."
        ):

            inventory = (
                generate_regulation_inventory(
                    regulatory_text
                )
            )

            st.session_state[
                "inventory"
            ] = inventory

            st.success(
                f"{len(inventory)} requirements extracted"
            )

            st.json(inventory)

# ---------------------------------
# SUBMISSION FILE
# ---------------------------------

submission_file = st.file_uploader(
    "Upload Submission",
    type=["docx", "pdf", "txt"]
)

if (
    submission_file
    and "inventory" in st.session_state
):

    if st.button(
        "Run Compliance Review"
    ):

        with st.spinner(
            "Reviewing submission..."
        ):

            suffix = "." + submission_file.name.split(".")[-1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(
                    submission_file.read()
                )

                submission_path = tmp.name

            submission_text = load_document(
                submission_path
            )

            submission_chunks = chunk_text(
                submission_text,
                chunk_size=300
            )

            submission_index = (
                build_faiss_index(
                    submission_chunks
                )
            )

            findings = (
                run_compliance_review(
                    st.session_state["inventory"],
                    submission_chunks,
                    submission_index
                )
            )

            st.subheader(
                "Compliance Findings"
            )

            st.dataframe(
                findings,
                use_container_width=True
            )

            report = export_to_excel(
                findings
            )

            with open(
                report,
                "rb"
            ) as f:

                st.download_button(
                    "Download Excel Report",
                    f,
                    file_name="Compliance_Report.xlsx"
                )