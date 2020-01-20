# python script to
# 1. download all data .gz files from GCS
# 2. decompress files and save locally as csv
# 3. read csv files into dataframe and check for errors
# 4. upload csv to GCS
# 5. upload dataframe to Bigquery

# ===================library imports===================#
# GCP client library imports
from google.cloud import storage
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField as SF
from google.cloud.bigquery import TimePartitioning as TP

# file/system imports
import os
import gzip
import shutil
from pathlib import Path
import csv
import json
import logging
import sys
from multiprocessing import Pool

# dataframe imports
import pandas_gbq
import pandas as pd
import numpy as np
from dask.distributed import Client as dClient
import dask.dataframe as dd

# string processing imports
import re
from fuzzywuzzy import process, fuzz

# =====================================================#

# ------to handle large csv fields--------#
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)
# ----------------------------------------#

# ===================set up ====================#
# Instantiates a GCP storage client
storage_client = storage.Client()
bq_client = bigquery.Client()

# define general GCP parameters
project_id = "gum-eroski-dev"

# define read GCP parameters
dataset_id = "source_data"
bucket = "erk-data-feed"
storage_filepath = "eroski-deltas/AVANTE_INCR" #"eroski-deltas"
blobs = storage_client.list_blobs(bucket, prefix=storage_filepath)
blob_list = [blob.name for blob in blobs]

# define write GCP parameters
write_dataset_id = "delta_data_secondbatch"
dataset_ref = bq_client.dataset(write_dataset_id)
write_storage_filepath = "Working_folder/SL/ETL_test_upload/"



bad_rows_allowed = 0.05  # percentage of bad rows allowed in csv to write to bq

# define compute local disk file directory locations
home = str(Path.home())
local_dir = os.path.abspath(home + "/etl_test/")

# open table_mapping.json for delta to bq table mapping
with open("table_mapping.json", "r") as mapping:
    table_mapping = json.load(mapping)
# =====================================================#

# ===================function definitions====================#


def bq_write(fpath, table_id: str, header: int, table_dtypes: dict, original_row_count):
    """Write to bigquery"""
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True
    job_config.write_disposition = "WRITE_TRUNCATE"
    job_config.skip_leading_rows = header
    job_config.allow_jagged_rows = True
    job_config.field_delimiter = "|"
    job_config.max_bad_records = int(original_row_count * bad_rows_allowed)

    # set write schema using original bq table
    schema = []
    for i in table_dtypes:
        schema.append(SF(i, table_dtypes[i]))
    job_config.schema = schema

    # write file to bigquery
    with open(fpath, "rb") as source_file:
        job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result()  # Waits for table load to complete.


def bq_add_timestamp(table_id, timestamp):
    """Query bigquery and add timestamp"""
    table_ref = project_id + "." + write_dataset_id + "." + table_id
    job_config = bigquery.QueryJobConfig()
    job_config.destination = table_ref
    job_config.dry_run = False
    job_config.write_disposition = "WRITE_APPEND"
    # job_config.create_disposition = "CREATE_NEVER"
    job_config.use_query_cache = True
    # job_config.time_partitioning = TP(type_="DAY", field="TIMESTAMP")

    sql = """
        SELECT *, {t} as TIMESTAMP
        FROM `{table}`;
    """.format(
        t=timestamp, table=table_ref + "_delta"
    )

    # Start the query, passing in the extra configuration.
    query_job = bq_client.query(sql, job_config=job_config)  # Make an API request.
    query_job.result()  # Wait for the job to complete.

    logger.info("Timestamp added to bq table {}".format(table_id))


def bq_get_summary_stats(table_id):
    """Return summary statistics of bigquery table"""
    table_ref = project_id + "." + write_dataset_id + "." + table_id
    job_config = bigquery.QueryJobConfig()
    job_config.use_query_cache = True

    sql = """
        WITH `table` AS (
          SELECT * FROM `{table}`
        ),
        table_as_json AS (
          SELECT  REGEXP_REPLACE(TO_JSON_STRING(t), r'^{{|}}$', '') AS row
          FROM `table` AS t
        ),
        pairs AS (
          SELECT
            REPLACE(column_name, '"', '') AS column_name,
            IF(SAFE_CAST(column_value AS STRING)='null',NULL,column_value) AS column_value,
            IF(SAFE_CAST(column_value AS numeric) is null, NULL, SAFE_CAST(column_value AS numeric)) AS col_nu
          FROM table_as_json, UNNEST(SPLIT(row, ',"')) AS z,
          UNNEST([SPLIT(z, ':')[SAFE_OFFSET(0)]]) AS column_name,
          UNNEST([SPLIT(z, ':')[SAFE_OFFSET(1)]]) AS column_value
        )
        SELECT
          column_name,
          COUNT(DISTINCT column_value) AS distinct_values,
          COUNTIF(column_value IS NULL) AS no_nulls,
          COUNTIF(column_value IS NOT NULL) AS no_non_nulls,
          ROUND(AVG(col_nu),1) AS avg_value,
          MAX(col_nu) AS max_value,
          MIN(col_nu) AS min_value
        FROM pairs
        WHERE column_name <> ''
        GROUP BY column_name
        ORDER BY column_name
    """.format(
        table=table_ref
    )

    # Start the query, passing in the extra configuration.
    query_job = bq_client.query(sql, job_config=job_config)  # Make an API request.
    return query_job.result().to_dataframe()  # return the dataframe result


def bq_get_row_count(table_id):
    """Return row count of bigquery table"""
    table_ref = project_id + "." + write_dataset_id + "." + table_id
    job_config = bigquery.QueryJobConfig()
    job_config.use_query_cache = True

    sql = """
        SELECT COUNT(*)
        FROM `{table}`;
    """.format(
        table=table_ref
    )

    # Start the query, passing in the extra configuration.
    query_job = bq_client.query(sql, job_config=job_config)  # Make an API request.
    return query_job.result().to_dataframe().f0_.iloc[0]  # return the first result (count of rows)


def initialise_logger():
    """Initialise logger settings"""

    # Set logger properties
    logger = logging.getLogger("auto_etl_to_bq")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("auto_etl_to_bq.log")
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def download_blob(bucket_name, source_blob_name, destination_file_name, replace=False):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    # download blob depending on replace flag
    if os.path.exists(destination_file_name) and replace:
        logger.info(
            "Blob {} already exists in {}...overwriting.".format(blob, destination_file_name)
        )
        blob.download_to_filename(destination_file_name)
    elif os.path.exists(destination_file_name) and replace == False:
        logger.info(
            "Blob {} already exists in {}...skipping download.".format(blob, destination_file_name)
        )
    else:
        blob.download_to_filename(destination_file_name)
        logger.info("Blob {} downloaded to {}".format(source_blob_name, destination_file_name))


def gunzip(source_filepath, dest_filepath, block_size=65536):
    """Unzips .gz files and writes to persistent disk"""

    with gzip.open(source_filepath, "rb") as s_file, open(dest_filepath, "wb") as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)
        d_file.write(block)


def upload_blob(bucket_name, source_file_name, destination_blob_name, replace=False):
    """Uploads a file to the bucket"""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # upload file to GCS depending on replace flag
    if blob.exists() and replace:
        logger.info("Overwriting GCS blob {b}".format(b=destination_blob_name))
        blob.upload_from_filename(source_file_name)
    elif blob.exists() and replace == False:
        logger.info("GCS blob {} exists already...skipping upload".format(destination_blob_name))
    else:
        blob.upload_from_filename(source_file_name)
        logger.info("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def change_extension(old_extension, new_extension, directory):
    """Change file extensions for files in directory"""

    for file in os.listdir(directory):
        pre, ext = os.path.splitext(file)
        if ext == old_extension:
            os.rename(file, pre + new_extension)
            continue
        else:
            continue


def csv_checks(csv_filename, dataset_schema):
    """Checks format of delta csv files with Bigquery tables"""

    logger.info(".........beginning checks for {}.........".format(csv_filename))
    # check csv file nrows
    csv_row_count = sum(1 for row in csv.reader(open(csv_filename)))
    # read top 5 lines of csv file into dataframe
    try:
        csv_data = pd.read_csv(
            csv_filename, header=None, index_col=False, sep="|", engine="python", nrows=5
        )
        # check ncolumns
        csv_column_count = len(csv_data.columns)
        logger.info("csv file: {} loaded to dataframe".format(csv_filename))
        read_successful = True
    except:
        logger.info("csv file: {} did not read properly".format(csv_filename))
        read_successful = False

    if read_successful:
        # check for matching table in Bigquery
        fn = csv_filename.split("/")[-1]
        table_name_list = dataset_schema.table_name.unique()

        # replace digits with X and remove extension
        fn_str = re.sub(r"\d", "X", fn.split(".")[0])

        # if file mapping exists for file name
        if table_mapping[fn_str] != "":
            # using mapping table select the correct schema
            matched_table_schema = dataset_schema.loc[
                dataset_schema.table_name == table_mapping[fn_str]
            ]
            # create dictionary columns:dtypes
            table_dtypes = {}
            for idx, col in enumerate(matched_table_schema.column_name.tolist()):
                table_dtypes[col] = matched_table_schema.data_type.tolist()[idx]
            # get first row of csv dataframe
            csv_header = list(csv_data.iloc[0])

            # get column names of bq table
            table_columns = matched_table_schema.column_name.tolist()

            # compare csv first row and bq table column names
            csv_header = [str(x).lower() for x in csv_header]
            table_columns_lower = [x.lower() for x in table_columns]
            if (
                len(csv_header) == len(table_columns_lower)
                and len(csv_header)
                == sum([1 for i, j in zip(csv_header, table_columns_lower) if i == j])
                or csv_data[csv_data.columns[0]].iloc[0] == csv_data.columns[0]
            ):

                logger.info(
                    "CSV header exists...writing {} to bigquery without first row".format(fn)
                )
                # write to bq without header row
                header_row = 1
                bq_write(
                    csv_filename,
                    table_mapping[fn_str] + "_delta",
                    header_row,
                    table_dtypes,
                    csv_row_count,
                )
                logger.info("Finished writing to bigquery")
            else:
                logger.info(
                    "CSV header does not exist...writing {} to bigquery applying original source data headers".format(
                        fn
                    )
                )
                header_row = 0
                bq_write(
                    csv_filename,
                    table_mapping[fn_str] + "_delta",
                    header_row,
                    table_dtypes,
                    csv_row_count,
                )

            logger.info("Adding timestamp column to table")
            bq_add_timestamp(table_mapping[fn_str], re.findall("\d+", fn)[0])

            # log final row count
            final_row_count = bq_get_row_count(table_mapping[fn_str] + "_delta") + header_row
            logger.info(
                "original csv file {f} has {n} rows".format(
                    f=csv_filename.split("/")[-1], n=csv_row_count
                )
            )
            logger.info("number of rows added = {r}".format(r=final_row_count))
            logger.info("number of error rows skipped = {}".format(csv_row_count - final_row_count))
            logger.info("retrieving summary stats.....")
            summary_stats = bq_get_summary_stats(table_mapping[fn_str] + "_delta")
            summary_stats.to_csv(
                os.path.abspath(local_dir + "/" + fn.split(".")[0] + "_summary_stats" + ".csv")
            )
            upload_blob(
                bucket,
                os.path.abspath(local_dir + "/" + fn.split(".")[0] + "_summary_stats" + ".csv"),
                write_storage_filepath + fn.split(".")[0] + "_summary_stats" + ".csv",
                replace=False,
            )
            logger.info(
                "Summary stats saved to {}".format(
                    write_storage_filepath + fn.split(".")[0] + "_summary_stats" + ".csv"
                )
            )
        else:
            logger.info("Delta table {} does not have mapping".format(fn))
    else:
        logger.info("Did not attempt to upload {} to Bigquery".format(csv_filename))


def get_bq_schemas(dataset_id):
    """Returns Bigquery dataset information"""

    # get table names and columns
    sql_str = """
    SELECT table_name, column_name, data_type FROM `gum-eroski-dev`.source_data.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS
    """
    # read from Bigquery
    dataset_schema = pandas_gbq.read_gbq(sql_str, project_id=project_id)
    return dataset_schema


if __name__ == "__main__":
    logger = initialise_logger()
    dataset_schema = get_bq_schemas(dataset_id)
    for blob in blob_list:
        blob_fn = blob.split("/")[-1]
        logger.info("-----------------Starting ETL of {}-----------------".format(blob_fn))
        download_blob(bucket, blob, os.path.abspath(local_dir + "/" + blob_fn), replace=False)
        while not os.path.exists(os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv")):
            if os.path.abspath(local_dir + "/" + blob_fn.split(".")[-1] != "gz":
                gunzip(
                os.path.abspath(local_dir + "/" + blob_fn),
                os.path.abspath(local_dir + "/" + blob_fn.split(".")[-1] + "_" + os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"),
                )
            else:
                gunzip(
                    os.path.abspath(local_dir + "/" + blob_fn),
                    os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv")
                )
            upload_blob(
                bucket,
                os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"),
                write_storage_filepath + blob_fn.split(".")[0] + ".csv",
                replace=False,
            )
        else:
            logger.info(
                "File {} already unzipped".format(os.path.abspath(local_dir + "/" + blob_fn))
            )
            csv_checks(
                os.path.abspath(local_dir + "/" + blob_fn.split(".")[0] + ".csv"), dataset_schema,
            )
        logger.info("-----------------Finished ETL of {}-----------------".format(blob_fn))
