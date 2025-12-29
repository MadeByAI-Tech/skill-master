from pipeline.phase1.function.extract_all_components import extract_all_components
from db import get_client, get_dataset
from google.cloud import bigquery

"""
Pseudo code

jobposts = JobPost.fetch_all()

preprocessed_jobposts = []

for JobPost in JobPosts:
    preprocessed_jobposts.append(preprocessing(JobPost))

extracted_jobposts = []

for JobPost in preprocessed_jobposts:
    extracted_jobposts.append(extraction(JobPost))

postporoocessed_jobposts = []

for JobPost in preprocessed_jobposts:
    postporoocessed_jobposts.append(postprocessing(JobPost))
"""


def run():
    """
    Pipeline to transform raw JobPosts into `JobPostExtracted` records
        - Fetches rows from the `JobPosts` table
        - Extracts structured fields using `extract_all_components()`
        - Inserts results into the `JobPostExtracted` table
    """

    print("─" * 30)
    print("     Pipeline started")
    print("─" * 30, flush=True)

    # Initalize BigQuery client
    print("=== RUNNING: get_client()...", flush=True)
    client = get_client()
    print("=== DONE:    get_client()", flush=True)

    # Load dataset
    print("=== RUNNING: get_dataset()...", flush=True)
    dataset = get_dataset()
    dataset_id = dataset.dataset_id
    print("=== DONE:    get_dataset()", flush=True)

    # Source and destination table
    JOBPOST_TABLE = f"{client.project}.{dataset_id}.JobPosts"
    EXTRACTED_TABLE = f"{dataset_id}.JobPostExtracted"

    # Read data from source table without using SQL
    print("=== Reading rows from JobPosts table...", flush=True)
    table = client.get_table(JOBPOST_TABLE)

    rows = list(client.list_rows(table, max_results=100))
    print(f"=== Row fetch completed. Total rows = {len(rows)}", flush=True)

   # Limit row for testing
    rows = rows[:5]
    print(f"=== Processing `{len(rows)}` rows for testing", flush=True)

    to_insert = []

    print("")
    print("─" * 30)
    print("     Extraction started")
    print("─" * 30)

    # Process rows and extract data
    for i, r in enumerate(rows):
        print(f"- Processing row {i+1}/{len(rows)}, job_id={r['job_id']}", flush=True)

        out = extract_all_components(
            job_title=r["job_title"] or "",
            job_function=r["job_function"] or "",
            job_description=r["description"] or ""
        )
        
        print(f"  Row {i+1} extraction completed", flush=True)

        to_insert.append({
            "job_id":       r["job_id"],
            "job_title":    r["job_title"],
            "location":     r["location"],
            "job_function": r["job_function"],
            "description":  r["description"],

            "skills":       out.get("skills"),
            "cwf_items":    out.get("cwf_items"),
        })

    # Insert processed records into destination table
    print("=== Inserting processed rows into JobPostExtracted...", flush=True)
    errors = client.insert_rows_json(EXTRACTED_TABLE, to_insert)

    if not errors:
        print("=== Data inserted successfully to JobPostExtracted table")
        return "200 OK"
    else:
        print("Error inserting data:", errors)
        return "404 ERROR"