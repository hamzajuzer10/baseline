import os
import json
from pathlib import Path
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField as SF
import unzip_file
import pandas as pd

client = bigquery.Client()

home = str(Path.home())
filename = os.path.abspath(home + "/etl_test/20191128_M_ARTICULOS_20191128.csv")
dataset_id = "WIP"
table_id = "upload_test"

dataset_ref = client.dataset(dataset_id)
table_ref = dataset_ref.table(table_id)
job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.CSV
job_config.autodetect = True
job_config.write_disposition = "WRITE_TRUNCATE"
job_config.skip_leading_rows = 1
job_config.field_delimiter = ","

# schema = [SF("col1", "STRING"), SF("two", "STRING"), SF("three", "STRING"), SF("four", "STRING")]
# schema = [SF(a, b), SF("two", "STRING"), SF("three", "STRING"), SF("four", "STRING")]

dataset_schema = unzip_file.get_bq_schemas("source_data")
matched_table_schema = dataset_schema.loc[
    dataset_schema.table_name == table_mapping["11_M_ARTICULOS"]
]
table_columns = matched_table_schema.column_name.tolist()

schema = []
for i in table_columns:
    schema.append(SF(i, "STRING"))

job_config.allow_jagged_rows = True
job_config.schema = schema

with open(filename, "rb") as source_file:
    job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

job.result()  # Waits for table load to complete.
