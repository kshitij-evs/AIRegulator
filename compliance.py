import json
import pandas as pd

from rag import (
    client,
    CHAT_MODEL,
    retrieve_relevant_chunks
)

# ------------------------
# Requirement Extraction
# ------------------------

def generate_regulation_inventory(
        regulatory_text):

    prompt = f"""
Extract all regulatory requirements.

Return ONLY JSON.

Format:

[
 {{
   "requirement_id":"REQ-001",
   "category":"Documentation",
   "requirement":"Description",
   "severity_if_missing":"High"
 }}
]

Regulatory Text:

{regulatory_text}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role":"system",
                "content":"Regulatory expert."
            },
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    output = response.choices[0].message.content

    return json.loads(output)


# ------------------------
# Requirement Assessment
# ------------------------

def assess_requirement(
        requirement,
        evidence):

    prompt = f"""
Requirement:

{requirement["requirement"]}

Submission Evidence:

{evidence}

Evaluate compliance.

Return ONLY valid JSON:

{{
 "requirement_id":"",
 "category":"",
 "status":"Compliant|Partial|Missing|Non-Compliant",
 "severity":"",
 "issue":"",
 "evidence":"",
 "remediation":""
}}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role":"system",
                "content":"Compliance reviewer"
            },
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )


# ------------------------
# Compliance Review
# ------------------------

def run_compliance_review(
        inventory,
        submission_chunks,
        submission_index):

    findings = []

    total = len(inventory)

    for count, req in enumerate(
            inventory,
            start=1):

        print(
            f"Checking {count}/{total}"
        )

        evidence = retrieve_relevant_chunks(
            req["requirement"],
            submission_chunks,
            submission_index
        )

        result = assess_requirement(
            req,
            evidence
        )

        findings.append(result)

    return findings


# ------------------------
# Excel Export
# ------------------------

def export_to_excel(
        findings,
        file_name="Compliance_Report.xlsx"):

    df = pd.DataFrame(findings)

    df.to_excel(
        file_name,
        index=False,
        engine="openpyxl"
    )

    return file_name