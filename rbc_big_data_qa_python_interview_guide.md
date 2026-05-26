# Python Interview Guide for Big Data QA Roles at RBC

**Target role:** Big Data QA / Quality Engineer / Data Quality Engineer / Automation QA Engineer  
**Primary focus:** Python scripting, data validation, pandas, PySpark basics, file processing, SQL-adjacent validation, automation, logging, testing, and production-support thinking  
**Audience:** Candidate preparing for RBC-style QA interviews in banking/financial data environments  
**Status:** Interview-prep guide, not an official RBC document  
**Last updated:** 2026-05-25

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)
2. [What RBC-Style Big Data QA Interviews Usually Test](#2-what-rbc-style-big-data-qa-interviews-usually-test)
3. [Python Skills Map for Big Data QA](#3-python-skills-map-for-big-data-qa)
4. [Python Fundamentals You Must Know](#4-python-fundamentals-you-must-know)
5. [Working With Files and Directories](#5-working-with-files-and-directories)
6. [CSV and Delimited File Processing](#6-csv-and-delimited-file-processing)
7. [pandas for Data QA](#7-pandas-for-data-qa)
8. [Large File Handling and Chunk Processing](#8-large-file-handling-and-chunk-processing)
9. [Data Quality Validation Patterns](#9-data-quality-validation-patterns)
10. [Source-to-Target Reconciliation](#10-source-to-target-reconciliation)
11. [Python With SQL and Databases](#11-python-with-sql-and-databases)
12. [PySpark Basics for Big Data QA](#12-pyspark-basics-for-big-data-qa)
13. [JSON, Parquet, and Other Data Formats](#13-json-parquet-and-other-data-formats)
14. [Logging, Error Handling, and Exit Codes](#14-logging-error-handling-and-exit-codes)
15. [Command-Line Python Scripts With argparse](#15-command-line-python-scripts-with-argparse)
16. [pytest and unittest for QA Automation](#16-pytest-and-unittest-for-qa-automation)
17. [Reusable Validation Framework Design](#17-reusable-validation-framework-design)
18. [Banking and RBC-Style Data QA Mindset](#18-banking-and-rbc-style-data-qa-mindset)
19. [Common Python Interview Questions and Answers](#19-common-python-interview-questions-and-answers)
20. [Hands-On Coding Exercises](#20-hands-on-coding-exercises)
21. [Mini Project: Python Data QA Validation Framework](#21-mini-project-python-data-qa-validation-framework)
22. [Mock Interview Round](#22-mock-interview-round)
23. [10-Day Preparation Plan](#23-10-day-preparation-plan)
24. [Final Cheat Sheet](#24-final-cheat-sheet)
25. [Recommended Practice Datasets](#25-recommended-practice-datasets)
26. [References](#26-references)

---

# 1. How to Use This Guide

This guide is designed to help you prepare for Python questions in a **Big Data QA** or **Quality Engineer** interview at a financial institution such as RBC.

Do not prepare Python only as a general programming language. For a Big Data QA role, you should connect Python to real QA work:

- Reading and validating files.
- Comparing source and target datasets.
- Checking duplicates, nulls, schema, data types, and record counts.
- Automating daily validation jobs.
- Reading logs and generating QA reports.
- Using pandas for small-to-medium files.
- Understanding PySpark for distributed data validation.
- Writing reusable scripts with clear logging and error handling.

A strong interview answer usually has three parts:

1. **Python knowledge**  
   Example: lists, dictionaries, functions, file handling, exceptions, modules, pandas.

2. **QA/data validation thinking**  
   Example: source count equals target count, mandatory fields are not null, duplicate primary keys are detected.

3. **Production mindset**  
   Example: logs, exit codes, configuration files, masking sensitive data, auditability, retry strategy, and failure reporting.

---

# 2. What RBC-Style Big Data QA Interviews Usually Test

Public RBC Quality Engineer and data engineering postings commonly emphasize skills such as:

- Unix/Linux environments.
- Python scripting.
- SQL queries.
- Big Data/Hadoop.
- Spark/PySpark exposure.
- Cloud or distributed data platforms.
- Automated testing.
- Production troubleshooting.
- Quality engineering for cross-platform systems.

For a Python-focused Big Data QA interview, you may be asked to show that you can:

- Write Python scripts for data validation.
- Use pandas to inspect and compare files.
- Use PySpark concepts for large-scale data.
- Read CSV, JSON, and possibly Parquet files.
- Validate record counts, duplicates, nulls, and business rules.
- Automate reports and produce pass/fail output.
- Explain how your script behaves when something fails.
- Write test cases using `pytest` or `unittest`.
- Work with database query outputs.

## What the interviewer wants to know

They may ask:

- Can you write clean Python, not just copy commands?
- Can you debug data issues logically?
- Can you explain why a validation failed?
- Can you process large files safely?
- Can you avoid exposing sensitive banking data in logs?
- Can you convert a manual QA check into an automated script?

---

# 3. Python Skills Map for Big Data QA

## Must-have Python skills

| Skill | Why it matters in Big Data QA |
|---|---|
| Variables and data types | Store counts, paths, thresholds, statuses |
| Lists, tuples, sets, dictionaries | Handle records, unique IDs, mappings, validation rules |
| Functions | Make validation logic reusable |
| File handling | Read and write input/output files |
| CSV handling | Validate delimited files from ETL jobs |
| pandas | Analyze tabular datasets quickly |
| Exceptions | Fail gracefully and explain errors |
| Logging | Create audit-friendly validation logs |
| argparse | Build reusable command-line validation tools |
| pytest/unittest | Automate test cases |
| SQL integration | Validate database extracts and query results |
| PySpark basics | Understand validation on distributed datasets |

## Nice-to-have skills

| Skill | Why it helps |
|---|---|
| Type hints | Makes scripts easier to understand |
| Dataclasses | Useful for configuration and result objects |
| pathlib | Cleaner path handling |
| configparser/YAML/JSON configs | Separates rules from code |
| Great Expectations or similar tools | Data quality framework awareness |
| Airflow awareness | Useful for scheduled pipeline validation |
| Git/Jenkins integration | Helps with CI/CD testing workflows |

---

# 4. Python Fundamentals You Must Know

## Variables

```python
file_name = "customer_data.csv"
expected_count = 1000
is_valid = True
```

## Common data types

```python
name = "Harsh"              # string
count = 100                 # integer
amount = 250.75             # float
is_active = True            # boolean
ids = [101, 102, 103]       # list
unique_ids = {101, 102}     # set
record = {"id": 101, "status": "ACTIVE"}  # dictionary
```

## Lists

```python
files = ["customer.csv", "account.csv", "transaction.csv"]

for file in files:
    print(file)
```

## Dictionaries

Dictionaries are very common in QA automation because they store mappings and validation rules.

```python
expected_schema = {
    "customer_id": "string",
    "customer_name": "string",
    "account_status": "string",
    "balance": "float"
}
```

## Sets

Sets are useful for duplicate detection and comparing source/target keys.

```python
source_ids = {101, 102, 103}
target_ids = {101, 102, 104}

missing_in_target = source_ids - target_ids
extra_in_target = target_ids - source_ids

print(missing_in_target)  # {103}
print(extra_in_target)    # {104}
```

## Functions

```python
def count_records(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file) - 1  # excluding header

count = count_records("customer.csv")
print(count)
```

## Interview explanation

A strong answer:

> I use functions to keep validation logic reusable. For example, I would create separate functions for file existence, record count validation, duplicate checks, null checks, and report generation. This makes the framework easier to maintain and extend.

---

# 5. Working With Files and Directories

Python is frequently used to automate file checks before or after batch processing.

## Check if a file exists

```python
from pathlib import Path

file_path = Path("customer_data.csv")

if file_path.exists():
    print("File exists")
else:
    print("File does not exist")
```

## Check if file is empty

```python
from pathlib import Path

file_path = Path("customer_data.csv")

if file_path.stat().st_size == 0:
    print("File is empty")
else:
    print("File is not empty")
```

## List files in a directory

```python
from pathlib import Path

folder = Path("/data/input")

for file in folder.glob("*.csv"):
    print(file.name)
```

## Find latest file

```python
from pathlib import Path

folder = Path("/data/input")
csv_files = list(folder.glob("*.csv"))

latest_file = max(csv_files, key=lambda file: file.stat().st_mtime)
print(latest_file)
```

## Common interview question

**Q: How would you validate that a file arrived before processing starts?**

Strong answer:

> I would check whether the file exists, whether it is non-empty, whether its naming convention matches the expected pattern, whether the modified timestamp is within the expected batch window, and whether its header/trailer matches the agreed format. I would log the result and stop the pipeline if a mandatory input file is missing.

Example:

```python
from pathlib import Path
import re

file_path = Path("customer_20260525.csv")
pattern = r"^customer_\d{8}\.csv$"

if not file_path.exists():
    raise FileNotFoundError(f"Missing file: {file_path}")

if file_path.stat().st_size == 0:
    raise ValueError(f"Empty file: {file_path}")

if not re.match(pattern, file_path.name):
    raise ValueError(f"Invalid file name: {file_path.name}")

print("File arrival validation passed")
```

---

# 6. CSV and Delimited File Processing

Big Data QA often involves CSV, pipe-delimited, tab-delimited, fixed-width, or control files.

Python has a built-in `csv` module for reading and writing CSV files.

## Read a CSV file using the built-in csv module

```python
import csv

with open("customer.csv", "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        print(row)
```

## Read CSV as dictionaries

```python
import csv

with open("customer.csv", "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        print(row["customer_id"], row["account_status"])
```

## Count records excluding header

```python
import csv

record_count = 0

with open("customer.csv", "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for _ in reader:
        record_count += 1

print(record_count)
```

## Validate column count

```python
import csv

expected_columns = 5
bad_rows = []

with open("customer.csv", "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file)

    for line_number, row in enumerate(reader, start=1):
        if len(row) != expected_columns:
            bad_rows.append((line_number, row))

if bad_rows:
    print(f"FAIL: {len(bad_rows)} rows have invalid column count")
else:
    print("PASS: Column count is valid")
```

## Read pipe-delimited file

```python
import csv

with open("transactions.psv", "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file, delimiter="|")

    for row in reader:
        print(row)
```

## Common interview question

**Q: Why not just use `split(',')` for CSV files?**

Strong answer:

> Simple `split(',')` can fail when fields contain commas inside quotes, embedded delimiters, or special quoting rules. The Python `csv` module handles standard CSV parsing more safely.

---

# 7. pandas for Data QA

`pandas` is one of the most useful Python libraries for QA validation on small-to-medium tabular datasets. It is especially useful for CSV extracts, reconciliation files, and quick exploratory checks.

## Read a CSV file

```python
import pandas as pd

df = pd.read_csv("customer.csv")
print(df.head())
```

## Basic inspection

```python
print(df.shape)       # rows and columns
print(df.columns)     # column names
print(df.dtypes)      # data types
print(df.info())      # summary
print(df.describe())  # numeric summary
```

## Count records

```python
record_count = len(df)
print(record_count)
```

## Check null values

```python
null_counts = df.isnull().sum()
print(null_counts)
```

## Check duplicates

```python
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)
```

## Check duplicate primary keys

```python
duplicate_customer_ids = df[df.duplicated(subset=["customer_id"], keep=False)]
print(duplicate_customer_ids)
```

## Validate mandatory fields

```python
mandatory_columns = ["customer_id", "account_status", "customer_name"]

for column in mandatory_columns:
    null_count = df[column].isnull().sum()
    blank_count = (df[column].astype(str).str.strip() == "").sum()
    print(f"{column}: null={null_count}, blank={blank_count}")
```

## Validate allowed values

```python
allowed_statuses = {"ACTIVE", "INACTIVE", "CLOSED"}

invalid_status_rows = df[~df["account_status"].isin(allowed_statuses)]

if invalid_status_rows.empty:
    print("PASS: All account statuses are valid")
else:
    print("FAIL: Invalid account statuses found")
    print(invalid_status_rows)
```

## Validate numeric range

```python
invalid_balance_rows = df[df["balance"] < 0]
print(invalid_balance_rows)
```

## Validate date format

```python
import pandas as pd

df["trade_date_parsed"] = pd.to_datetime(df["trade_date"], errors="coerce", format="%Y-%m-%d")
invalid_dates = df[df["trade_date_parsed"].isnull()]

print(invalid_dates)
```

## Compare two files with pandas

```python
import pandas as pd

source = pd.read_csv("source_customer.csv")
target = pd.read_csv("target_customer.csv")

source_ids = set(source["customer_id"])
target_ids = set(target["customer_id"])

missing_in_target = source_ids - target_ids
extra_in_target = target_ids - source_ids

print("Missing in target:", missing_in_target)
print("Extra in target:", extra_in_target)
```

## Important pandas interview point

A strong answer:

> pandas is useful for data QA when files fit in memory. For very large files, I would use chunking in pandas or move the validation to PySpark, Hive, or database-level SQL.

---

# 8. Large File Handling and Chunk Processing

Big Data QA often involves large files. You should not always load the full file into memory.

## Bad approach for huge files

```python
import pandas as pd

df = pd.read_csv("huge_file.csv")
```

This can fail if the file is too large for memory.

## Better approach: pandas chunking

```python
import pandas as pd

total_count = 0
null_customer_id_count = 0

for chunk in pd.read_csv("huge_file.csv", chunksize=100_000):
    total_count += len(chunk)
    null_customer_id_count += chunk["customer_id"].isnull().sum()

print("Total records:", total_count)
print("Null customer IDs:", null_customer_id_count)
```

## Find duplicates in chunks

Duplicate checks are harder with chunks because duplicate keys may appear in different chunks. You can use a set for manageable key volumes.

```python
import pandas as pd

seen_ids = set()
duplicate_ids = set()

for chunk in pd.read_csv("customer.csv", chunksize=100_000):
    for customer_id in chunk["customer_id"]:
        if customer_id in seen_ids:
            duplicate_ids.add(customer_id)
        else:
            seen_ids.add(customer_id)

print("Duplicate customer IDs:", duplicate_ids)
```

## Memory-friendly line count

```python
def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)

print(count_lines("large_file.csv"))
```

## Interview answer

**Q: How would you process a 20 GB file in Python?**

Strong answer:

> I would not load the entire file into memory. I would process it in chunks using pandas `chunksize`, stream it line by line using Python file handling, or use PySpark if the validation needs distributed processing. I would also push some checks to SQL/Hive/Spark when possible, especially count, duplicate, and aggregation validations.

---

# 9. Data Quality Validation Patterns

These are the most common checks for Big Data QA.

## 9.1 File-level checks

| Check | Purpose |
|---|---|
| File exists | Confirm source arrived |
| File is non-empty | Avoid processing empty data |
| File name format | Validate batch date and naming standard |
| File size | Detect abnormal extract size |
| Header exists | Validate structure |
| Trailer exists | Validate control totals |
| Encoding is valid | Avoid parsing failures |
| Delimiter is correct | Avoid shifted columns |

## File-level validation script

```python
from pathlib import Path
import re


def validate_file(file_path: str, expected_pattern: str) -> list[str]:
    errors = []
    path = Path(file_path)

    if not path.exists():
        errors.append(f"File does not exist: {file_path}")
        return errors

    if path.stat().st_size == 0:
        errors.append(f"File is empty: {file_path}")

    if not re.match(expected_pattern, path.name):
        errors.append(f"Invalid file name: {path.name}")

    return errors


errors = validate_file("customer_20260525.csv", r"^customer_\d{8}\.csv$")

if errors:
    print("FAIL", errors)
else:
    print("PASS")
```

## 9.2 Schema checks

```python
import pandas as pd

expected_columns = ["customer_id", "customer_name", "account_status", "balance"]
df = pd.read_csv("customer.csv")

actual_columns = list(df.columns)

if actual_columns == expected_columns:
    print("PASS: Schema matched")
else:
    print("FAIL: Schema mismatch")
    print("Expected:", expected_columns)
    print("Actual:", actual_columns)
```

## 9.3 Data type checks

```python
import pandas as pd

df = pd.read_csv("customer.csv")

df["balance_numeric"] = pd.to_numeric(df["balance"], errors="coerce")
invalid_balance = df[df["balance_numeric"].isnull()]

print(invalid_balance)
```

## 9.4 Null checks

```python
mandatory_columns = ["customer_id", "account_status"]

for column in mandatory_columns:
    invalid_rows = df[df[column].isnull() | (df[column].astype(str).str.strip() == "")]
    print(column, len(invalid_rows))
```

## 9.5 Duplicate checks

```python
duplicate_keys = df[df.duplicated(subset=["customer_id"], keep=False)]
print(duplicate_keys)
```

## 9.6 Business rule checks

```python
invalid_rows = df[(df["account_status"] == "CLOSED") & (df["balance"] > 0)]
print(invalid_rows)
```

Business explanation:

> If an account is closed, the business may expect the balance to be zero. If balance is greater than zero, we flag it for review.

---

# 10. Source-to-Target Reconciliation

Source-to-target reconciliation is one of the most important QA responsibilities in ETL and Big Data pipelines.

## Count reconciliation

```python
import pandas as pd

source = pd.read_csv("source.csv")
target = pd.read_csv("target.csv")

if len(source) == len(target):
    print("PASS: Counts match")
else:
    print(f"FAIL: Source={len(source)}, Target={len(target)}")
```

## Sum reconciliation

```python
source_total = source["transaction_amount"].sum()
target_total = target["transaction_amount"].sum()

if round(source_total, 2) == round(target_total, 2):
    print("PASS: Amount totals match")
else:
    print(f"FAIL: Source total={source_total}, Target total={target_total}")
```

## Key reconciliation

```python
source_keys = set(source["transaction_id"])
target_keys = set(target["transaction_id"])

missing_in_target = source_keys - target_keys
extra_in_target = target_keys - source_keys

print("Missing in target:", missing_in_target)
print("Extra in target:", extra_in_target)
```

## Column-level comparison

```python
merged = source.merge(
    target,
    on="transaction_id",
    how="inner",
    suffixes=("_src", "_tgt")
)

mismatches = merged[merged["transaction_amount_src"] != merged["transaction_amount_tgt"]]
print(mismatches)
```

## Hash-based reconciliation

Hashing is useful when comparing many columns.

```python
import pandas as pd
import hashlib


def row_hash(row, columns):
    combined = "|".join(str(row[column]) for column in columns)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

columns_to_compare = ["customer_id", "account_status", "balance"]

source["row_hash"] = source.apply(lambda row: row_hash(row, columns_to_compare), axis=1)
target["row_hash"] = target.apply(lambda row: row_hash(row, columns_to_compare), axis=1)

comparison = source[["customer_id", "row_hash"]].merge(
    target[["customer_id", "row_hash"]],
    on="customer_id",
    how="inner",
    suffixes=("_src", "_tgt")
)

mismatches = comparison[comparison["row_hash_src"] != comparison["row_hash_tgt"]]
print(mismatches)
```

## Interview answer

**Q: How do you validate that data moved correctly from source to target?**

Strong answer:

> I would start with record count reconciliation, then compare key counts, duplicate counts, null counts, and aggregate totals such as sum, min, and max. After that, I would perform field-level checks using business rules or hash-based comparison for selected columns. I would log mismatches and provide a summary report with pass/fail status.

---

# 11. Python With SQL and Databases

In a Big Data QA role, Python often works with SQL outputs.

You may use Python to:

- Run SQL queries.
- Compare query results.
- Export query results to CSV.
- Validate source and target tables.
- Generate reconciliation reports.

## Basic example with sqlite3

`sqlite3` is built into Python and useful for interview practice.

```python
import sqlite3

connection = sqlite3.connect("qa_demo.db")
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM customer")
count = cursor.fetchone()[0]

print(count)

connection.close()
```

## Compare two table counts

```python
import sqlite3

connection = sqlite3.connect("qa_demo.db")
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM source_customer")
source_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM target_customer")
target_count = cursor.fetchone()[0]

if source_count == target_count:
    print("PASS: Table counts match")
else:
    print(f"FAIL: Source={source_count}, Target={target_count}")

connection.close()
```

## Example with pandas and SQL

```python
import sqlite3
import pandas as pd

connection = sqlite3.connect("qa_demo.db")

df = pd.read_sql_query("SELECT * FROM customer", connection)
print(df.head())

connection.close()
```

## Interview answer

**Q: How do Python and SQL work together in QA?**

Strong answer:

> SQL is best for querying large database tables and doing aggregations close to the data. Python is useful for orchestration, parameterizing queries, comparing outputs, generating reports, and integrating validations into automation frameworks.

---

# 12. PySpark Basics for Big Data QA

PySpark is the Python API for Apache Spark. It is important for large-scale data validation because pandas may not be enough for very large datasets.

## When to use pandas vs PySpark

| Scenario | Better tool |
|---|---|
| Small CSV file | pandas |
| Quick local analysis | pandas |
| Multi-GB or TB data | PySpark |
| Distributed HDFS/S3 data | PySpark |
| Cluster-based transformation validation | PySpark |
| Simple local script | pandas or plain Python |

## Create SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("BigDataQAValidation") \
    .getOrCreate()
```

## Read CSV with PySpark

```python
df = spark.read.option("header", True).csv("/data/input/customer.csv")
df.show(5)
```

## Count records

```python
record_count = df.count()
print(record_count)
```

## Check schema

```python
df.printSchema()
```

## Check nulls

```python
from pyspark.sql.functions import col, count, when

null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

null_counts.show()
```

## Duplicate key check

```python
from pyspark.sql.functions import count

duplicates = df.groupBy("customer_id") \
    .agg(count("*").alias("record_count")) \
    .filter(col("record_count") > 1)

duplicates.show()
```

## Allowed value check

```python
allowed_statuses = ["ACTIVE", "INACTIVE", "CLOSED"]

invalid_statuses = df.filter(~col("account_status").isin(allowed_statuses))
invalid_statuses.show()
```

## Source-target count reconciliation

```python
source_df = spark.read.option("header", True).csv("/data/source/customer.csv")
target_df = spark.read.option("header", True).csv("/data/target/customer.csv")

source_count = source_df.count()
target_count = target_df.count()

if source_count == target_count:
    print("PASS: Counts match")
else:
    print(f"FAIL: Source={source_count}, Target={target_count}")
```

## Missing keys in target

```python
missing_in_target = source_df.select("customer_id").subtract(target_df.select("customer_id"))
missing_in_target.show()
```

## Interview answer

**Q: Why would you use PySpark instead of pandas?**

Strong answer:

> pandas works well for local in-memory analysis, but PySpark is better when the data is too large for one machine or stored in distributed systems like HDFS or cloud object storage. In Big Data QA, I would use PySpark for large-scale count checks, duplicate checks, null checks, schema validation, and source-to-target reconciliation.

---

# 13. JSON, Parquet, and Other Data Formats

Big Data environments often use more than CSV.

## JSON validation

```python
import json

with open("customer.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print(data)
```

## Validate required JSON fields

```python
required_fields = {"customer_id", "account_status", "created_date"}

for record in data:
    missing_fields = required_fields - set(record.keys())

    if missing_fields:
        print(f"Missing fields for record {record}: {missing_fields}")
```

## Read JSON with pandas

```python
import pandas as pd

df = pd.read_json("customer.json")
print(df.head())
```

## Read Parquet with pandas

```python
import pandas as pd

df = pd.read_parquet("customer.parquet")
print(df.head())
```

## Read Parquet with PySpark

```python
df = spark.read.parquet("/data/customer.parquet")
df.printSchema()
df.show(5)
```

## Interview answer

**Q: Why is Parquet common in Big Data?**

Strong answer:

> Parquet is a columnar storage format, so it is efficient for analytical queries and large-scale distributed processing. For QA, I would validate schema, row counts, nulls, duplicates, and aggregate values similarly to CSV, but using PySpark or tools that can read Parquet efficiently.

---

# 14. Logging, Error Handling, and Exit Codes

In production QA automation, `print()` is not enough. Use logging.

## Basic logging

```python
import logging

logging.basicConfig(
    filename="qa_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Validation started")
logging.warning("File size is lower than expected")
logging.error("Record count mismatch")
```

## Try/except example

```python
import logging
import pandas as pd

try:
    df = pd.read_csv("customer.csv")
    logging.info("File loaded successfully")
except FileNotFoundError:
    logging.error("Input file was not found")
    raise
except pd.errors.EmptyDataError:
    logging.error("Input file is empty")
    raise
except Exception as error:
    logging.exception("Unexpected error occurred")
    raise
```

## Exit codes

```python
import sys

validation_passed = False

if validation_passed:
    sys.exit(0)
else:
    sys.exit(1)
```

Exit code meaning:

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 or non-zero | Failure |

## Production-friendly validation script skeleton

```python
import logging
import sys
from pathlib import Path

logging.basicConfig(
    filename="qa_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_file(file_path: str) -> bool:
    path = Path(file_path)

    if not path.exists():
        logging.error("File does not exist: %s", file_path)
        return False

    if path.stat().st_size == 0:
        logging.error("File is empty: %s", file_path)
        return False

    logging.info("File validation passed: %s", file_path)
    return True


if __name__ == "__main__":
    result = validate_file("customer.csv")
    sys.exit(0 if result else 1)
```

## Important banking note

Avoid logging sensitive values such as:

- Full account numbers.
- Full customer IDs, if considered sensitive in the organization.
- Personal information.
- Full transaction details.
- Authentication tokens or passwords.

Better approach:

```python
logging.info("Validation failed for 12 records. See secure exception report.")
```

Instead of:

```python
logging.info("Customer 123456789 with account 987654321 failed validation")
```

---

# 15. Command-Line Python Scripts With argparse

In interviews, writing a reusable script is stronger than hardcoding file names.

## Basic argparse script

```python
import argparse

parser = argparse.ArgumentParser(description="Validate a CSV file")
parser.add_argument("--input", required=True, help="Path to input CSV file")
parser.add_argument("--expected-count", type=int, required=False, help="Expected record count")

args = parser.parse_args()

print(args.input)
print(args.expected_count)
```

Run:

```bash
python validate_file.py --input customer.csv --expected-count 1000
```

## Full count validation script

```python
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

logging.basicConfig(
    filename="qa_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_count(input_file: str, expected_count: int) -> bool:
    path = Path(input_file)

    if not path.exists():
        logging.error("Input file does not exist: %s", input_file)
        return False

    df = pd.read_csv(input_file)
    actual_count = len(df)

    logging.info("Expected count: %s", expected_count)
    logging.info("Actual count: %s", actual_count)

    if actual_count == expected_count:
        logging.info("PASS: Count matched")
        return True

    logging.error("FAIL: Count mismatch")
    return False


def main():
    parser = argparse.ArgumentParser(description="Record count validation")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--expected-count", required=True, type=int, help="Expected record count")
    args = parser.parse_args()

    passed = validate_count(args.input, args.expected_count)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

## Interview answer

**Q: Why use argparse?**

Strong answer:

> `argparse` makes the script reusable. Instead of hardcoding file paths or counts, I can pass them as command-line parameters. This allows the same validation script to run in dev, QA, or production with different inputs.

---

# 16. pytest and unittest for QA Automation

For a QA role, Python testing frameworks are important.

## Simple pytest test

```python
def add(a, b):
    return a + b


def test_add():
    assert add(2, 3) == 5
```

Run:

```bash
pytest test_sample.py
```

## Testing a validation function

Application code:

```python
def is_valid_status(status):
    return status in {"ACTIVE", "INACTIVE", "CLOSED"}
```

Test code:

```python
from validation import is_valid_status


def test_valid_status():
    assert is_valid_status("ACTIVE") is True


def test_invalid_status():
    assert is_valid_status("UNKNOWN") is False
```

## pytest parameterization

```python
import pytest
from validation import is_valid_status

@pytest.mark.parametrize(
    "status, expected",
    [
        ("ACTIVE", True),
        ("INACTIVE", True),
        ("CLOSED", True),
        ("UNKNOWN", False),
        ("", False),
    ]
)
def test_is_valid_status(status, expected):
    assert is_valid_status(status) == expected
```

## Temporary file test

```python
from pathlib import Path


def count_records(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file) - 1


def test_count_records(tmp_path):
    test_file = tmp_path / "customer.csv"
    test_file.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")

    assert count_records(test_file) == 2
```

## unittest example

```python
import unittest


def is_positive_amount(amount):
    return amount > 0


class TestAmountValidation(unittest.TestCase):
    def test_positive_amount(self):
        self.assertTrue(is_positive_amount(100))

    def test_negative_amount(self):
        self.assertFalse(is_positive_amount(-50))


if __name__ == "__main__":
    unittest.main()
```

## Interview answer

**Q: How do you test your Python validation scripts?**

Strong answer:

> I separate validation logic into functions and write unit tests for each rule. For example, I test valid and invalid account statuses, null checks, duplicate checks, and record count functions. I also create small sample input files to test pass and fail scenarios. This helps make the validation framework reliable and easier to maintain.

---

# 17. Reusable Validation Framework Design

A strong Python answer is not only about writing one script. Show that you can design reusable validation logic.

## Suggested project structure

```text
python_data_qa_framework/
│
├── config/
│   └── customer_rules.json
│
├── data/
│   ├── source_customer.csv
│   └── target_customer.csv
│
├── reports/
│   └── validation_report.csv
│
├── logs/
│   └── qa_validation.log
│
├── src/
│   ├── file_checks.py
│   ├── schema_checks.py
│   ├── data_quality_checks.py
│   ├── reconciliation.py
│   └── report_writer.py
│
├── tests/
│   └── test_data_quality_checks.py
│
└── run_validation.py
```

## Example config file

```json
{
  "file_name_pattern": "^customer_\\d{8}\\.csv$",
  "expected_columns": [
    "customer_id",
    "customer_name",
    "account_status",
    "balance"
  ],
  "mandatory_columns": [
    "customer_id",
    "account_status"
  ],
  "allowed_values": {
    "account_status": ["ACTIVE", "INACTIVE", "CLOSED"]
  },
  "primary_key": ["customer_id"]
}
```

## Validation result object

```python
from dataclasses import dataclass

@dataclass
class ValidationResult:
    check_name: str
    status: str
    actual_value: str
    expected_value: str
    message: str
```

## Example reusable check

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class ValidationResult:
    check_name: str
    status: str
    actual_value: str
    expected_value: str
    message: str


def check_record_count(df: pd.DataFrame, expected_count: int) -> ValidationResult:
    actual_count = len(df)
    status = "PASS" if actual_count == expected_count else "FAIL"

    return ValidationResult(
        check_name="record_count",
        status=status,
        actual_value=str(actual_count),
        expected_value=str(expected_count),
        message="Record count matched" if status == "PASS" else "Record count mismatch"
    )
```

## Writing report output

```python
import csv


def write_report(results, output_file):
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["check_name", "status", "actual_value", "expected_value", "message"]
        )
        writer.writeheader()

        for result in results:
            writer.writerow(result.__dict__)
```

## Interview answer

**Q: How would you design a Python automation framework for data QA?**

Strong answer:

> I would separate the framework into file checks, schema checks, data quality checks, reconciliation checks, configuration, logging, and reporting. Validation rules would be stored in a config file so that we do not need to change code for every dataset. Each validation would return a pass/fail result, and the framework would generate an audit-friendly report.

---

# 18. Banking and RBC-Style Data QA Mindset

For financial institutions, your answers should emphasize accuracy, traceability, security, and production discipline.

## Important points to mention

- Do not expose sensitive customer or account data in logs.
- Validate control totals for transaction files.
- Compare source and target counts.
- Validate monetary sums with correct rounding.
- Validate date rules such as trade date, settlement date, and processing date.
- Validate currency codes and account statuses.
- Keep reports audit-friendly.
- Fail fast for critical input issues.
- Clearly separate warnings from failures.
- Keep validation scripts reusable and configurable.

## Example banking validation rules

| Rule | Example validation |
|---|---|
| Mandatory customer ID | `customer_id` cannot be null |
| Unique transaction ID | No duplicate `transaction_id` |
| Valid amount | Amount must be numeric |
| Valid currency | Currency must be in allowed list |
| Valid date | Trade date must be valid date format |
| Reconciliation | Source sum equals target sum |
| Audit trail | Every validation run creates a report |
| Security | Logs do not reveal sensitive data |

## Strong interview phrase

> Since this is a banking data environment, I would make sure the Python validation is repeatable, auditable, secure, and produces clear evidence for pass/fail decisions without exposing sensitive information.

---

# 19. Common Python Interview Questions and Answers

## Q1. What is the difference between a list, tuple, set, and dictionary?

**Answer:**

- A list is ordered and mutable.
- A tuple is ordered and immutable.
- A set stores unique values and is useful for duplicate detection.
- A dictionary stores key-value pairs and is useful for mappings and validation rules.

Example:

```python
ids_list = [1, 2, 2, 3]
ids_set = set(ids_list)
print(ids_set)  # {1, 2, 3}
```

## Q2. How do you handle exceptions in Python?

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as error:
    print(f"Unexpected error: {error}")
```

Strong answer:

> In QA scripts, I use exceptions to handle missing files, empty files, invalid formats, and database connection failures. I log the error and return a non-zero exit code so the scheduler or CI/CD pipeline can detect failure.

## Q3. How do you count records in a CSV file?

```python
import pandas as pd

df = pd.read_csv("customer.csv")
print(len(df))
```

For a large file:

```python
count = 0
for chunk in pd.read_csv("customer.csv", chunksize=100_000):
    count += len(chunk)
print(count)
```

## Q4. How do you find duplicate customer IDs?

```python
duplicates = df[df.duplicated(subset=["customer_id"], keep=False)]
print(duplicates)
```

## Q5. How do you check null values?

```python
print(df.isnull().sum())
```

## Q6. How do you compare two files?

```python
source = pd.read_csv("source.csv")
target = pd.read_csv("target.csv")

source_keys = set(source["id"])
target_keys = set(target["id"])

print("Missing in target:", source_keys - target_keys)
print("Extra in target:", target_keys - source_keys)
```

## Q7. What is the difference between pandas and PySpark?

**Answer:**

pandas is mainly for local, in-memory data analysis. PySpark is used for distributed processing across a cluster and is better for very large datasets.

## Q8. What is a lambda function?

```python
square = lambda x: x * x
print(square(5))
```

Interview answer:

> A lambda is a small anonymous function. I use it occasionally for simple transformations, but for complex validation logic I prefer named functions because they are easier to read and test.

## Q9. What is list comprehension?

```python
numbers = [1, 2, 3, 4]
squares = [n * n for n in numbers]
print(squares)
```

## Q10. What are `*args` and `**kwargs`?

```python
def demo(*args, **kwargs):
    print(args)
    print(kwargs)

demo(1, 2, name="Harsh", role="QA")
```

Answer:

> `*args` accepts variable positional arguments, and `**kwargs` accepts variable keyword arguments.

## Q11. What is the use of `with open(...)`?

```python
with open("file.txt", "r", encoding="utf-8") as file:
    content = file.read()
```

Answer:

> It automatically closes the file after the block finishes, even if an error occurs.

## Q12. How would you make a Python script production-ready?

Strong answer:

> I would add command-line arguments, logging, exception handling, configuration files, clear exit codes, unit tests, and a validation report. I would also avoid logging sensitive data and make the script reusable across environments.

---

# 20. Hands-On Coding Exercises

## Exercise 1: File existence and empty check

Write a script that accepts a file path and checks whether the file exists and is not empty.

Solution:

```python
from pathlib import Path
import sys

file_path = Path(sys.argv[1])

if not file_path.exists():
    print("FAIL: File does not exist")
    sys.exit(1)

if file_path.stat().st_size == 0:
    print("FAIL: File is empty")
    sys.exit(1)

print("PASS: File exists and is not empty")
```

## Exercise 2: Count records excluding header

```python
import sys

file_path = sys.argv[1]

with open(file_path, "r", encoding="utf-8") as file:
    count = sum(1 for _ in file) - 1

print(f"Record count excluding header: {count}")
```

## Exercise 3: Validate column count

```python
import csv
import sys

file_path = sys.argv[1]
expected_columns = int(sys.argv[2])
bad_rows = []

with open(file_path, "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file)

    for line_number, row in enumerate(reader, start=1):
        if len(row) != expected_columns:
            bad_rows.append(line_number)

if bad_rows:
    print(f"FAIL: Invalid column count on rows: {bad_rows[:10]}")
    sys.exit(1)

print("PASS: Column count validation passed")
```

## Exercise 4: Find duplicate IDs

```python
import pandas as pd
import sys

file_path = sys.argv[1]
key_column = sys.argv[2]

df = pd.read_csv(file_path)
duplicates = df[df.duplicated(subset=[key_column], keep=False)]

if duplicates.empty:
    print("PASS: No duplicate keys")
else:
    print(f"FAIL: Found {len(duplicates)} duplicate rows")
    print(duplicates.head())
```

## Exercise 5: Check mandatory fields

```python
import pandas as pd
import sys

file_path = sys.argv[1]
mandatory_columns = sys.argv[2].split(",")

df = pd.read_csv(file_path)

for column in mandatory_columns:
    invalid_count = df[column].isnull().sum() + (df[column].astype(str).str.strip() == "").sum()
    print(f"{column}: {invalid_count} invalid values")
```

## Exercise 6: Generate a validation report

```python
import pandas as pd

file_path = "customer.csv"
df = pd.read_csv(file_path)

results = []

results.append({
    "check_name": "record_count",
    "status": "PASS" if len(df) > 0 else "FAIL",
    "actual_value": len(df),
    "expected_value": ">0"
})

results.append({
    "check_name": "duplicate_customer_id",
    "status": "PASS" if df.duplicated(subset=["customer_id"]).sum() == 0 else "FAIL",
    "actual_value": df.duplicated(subset=["customer_id"]).sum(),
    "expected_value": 0
})

report = pd.DataFrame(results)
report.to_csv("validation_report.csv", index=False)
print(report)
```

---

# 21. Mini Project: Python Data QA Validation Framework

This mini project is excellent interview preparation.

## Goal

Build a Python script that validates a customer file and produces a QA report.

## Input file

`customer_20260525.csv`

```csv
customer_id,customer_name,account_status,balance,currency
101,Alice,ACTIVE,100.50,CAD
102,Bob,CLOSED,0.00,CAD
103,Charlie,UNKNOWN,50.00,USD
104,,ACTIVE,25.75,CAD
101,Alice,ACTIVE,100.50,CAD
```

## Validation rules

1. File exists.
2. File is not empty.
3. File name matches `customer_YYYYMMDD.csv`.
4. Columns match expected schema.
5. `customer_id` cannot be null.
6. `customer_name` cannot be blank.
7. `account_status` must be one of `ACTIVE`, `INACTIVE`, `CLOSED`.
8. `customer_id` must be unique.
9. `balance` must be numeric.
10. `currency` must be one of `CAD`, `USD`.

## Full sample solution

```python
import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ValidationResult:
    check_name: str
    status: str
    actual_value: str
    expected_value: str
    message: str


logging.basicConfig(
    filename="qa_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


EXPECTED_COLUMNS = [
    "customer_id",
    "customer_name",
    "account_status",
    "balance",
    "currency"
]

ALLOWED_STATUSES = {"ACTIVE", "INACTIVE", "CLOSED"}
ALLOWED_CURRENCIES = {"CAD", "USD"}


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def validate_file_level(file_path: Path) -> list[ValidationResult]:
    results = []

    exists = file_path.exists()
    results.append(ValidationResult(
        "file_exists",
        pass_fail(exists),
        str(exists),
        "True",
        "File existence check"
    ))

    if not exists:
        return results

    non_empty = file_path.stat().st_size > 0
    results.append(ValidationResult(
        "file_not_empty",
        pass_fail(non_empty),
        str(file_path.stat().st_size),
        ">0 bytes",
        "File size check"
    ))

    pattern_match = bool(re.match(r"^customer_\d{8}\.csv$", file_path.name))
    results.append(ValidationResult(
        "file_name_pattern",
        pass_fail(pattern_match),
        file_path.name,
        "customer_YYYYMMDD.csv",
        "File naming convention check"
    ))

    return results


def validate_dataframe(df: pd.DataFrame) -> list[ValidationResult]:
    results = []

    schema_match = list(df.columns) == EXPECTED_COLUMNS
    results.append(ValidationResult(
        "schema_match",
        pass_fail(schema_match),
        str(list(df.columns)),
        str(EXPECTED_COLUMNS),
        "Schema validation"
    ))

    duplicate_count = df.duplicated(subset=["customer_id"]).sum()
    results.append(ValidationResult(
        "duplicate_customer_id",
        pass_fail(duplicate_count == 0),
        str(duplicate_count),
        "0",
        "Duplicate customer ID check"
    ))

    blank_name_count = (df["customer_name"].isnull() | (df["customer_name"].astype(str).str.strip() == "")).sum()
    results.append(ValidationResult(
        "blank_customer_name",
        pass_fail(blank_name_count == 0),
        str(blank_name_count),
        "0",
        "Mandatory customer name check"
    ))

    invalid_status_count = (~df["account_status"].isin(ALLOWED_STATUSES)).sum()
    results.append(ValidationResult(
        "invalid_account_status",
        pass_fail(invalid_status_count == 0),
        str(invalid_status_count),
        "0",
        "Allowed account status check"
    ))

    numeric_balance = pd.to_numeric(df["balance"], errors="coerce")
    invalid_balance_count = numeric_balance.isnull().sum()
    results.append(ValidationResult(
        "invalid_balance",
        pass_fail(invalid_balance_count == 0),
        str(invalid_balance_count),
        "0",
        "Numeric balance check"
    ))

    invalid_currency_count = (~df["currency"].isin(ALLOWED_CURRENCIES)).sum()
    results.append(ValidationResult(
        "invalid_currency",
        pass_fail(invalid_currency_count == 0),
        str(invalid_currency_count),
        "0",
        "Currency code check"
    ))

    return results


def write_report(results: list[ValidationResult], output_file: str) -> None:
    report = pd.DataFrame([result.__dict__ for result in results])
    report.to_csv(output_file, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer file QA validation")
    parser.add_argument("--input", required=True, help="Input customer CSV file")
    parser.add_argument("--report", default="validation_report.csv", help="Output report CSV file")
    args = parser.parse_args()

    file_path = Path(args.input)
    logging.info("Validation started for %s", file_path.name)

    results = validate_file_level(file_path)

    if any(result.status == "FAIL" for result in results):
        write_report(results, args.report)
        logging.error("File-level validation failed")
        return 1

    df = pd.read_csv(file_path)
    results.extend(validate_dataframe(df))
    write_report(results, args.report)

    failed_checks = [result for result in results if result.status == "FAIL"]

    if failed_checks:
        logging.error("Validation failed with %s failed checks", len(failed_checks))
        return 1

    logging.info("Validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Run the script

```bash
python customer_validation.py --input customer_20260525.csv --report customer_validation_report.csv
```

## How to explain this project in an interview

> I built a Python data QA validation script that checks file arrival, naming convention, schema, duplicates, mandatory fields, allowed values, and numeric fields. It generates a CSV report with pass/fail status for each check, uses logging for auditability, and returns exit code 1 if a critical validation fails. This pattern can be extended for other datasets by moving rules into configuration files.

---

# 22. Mock Interview Round

## Question 1

**Interviewer:** How would you validate a daily transaction file using Python?

**Strong answer:**

> I would first validate the file-level checks: whether the file exists, whether it is non-empty, whether the naming convention includes the correct batch date, and whether the delimiter/header are correct. Then I would use pandas or PySpark depending on file size to validate record count, mandatory fields, duplicate transaction IDs, amount fields, date formats, and currency codes. Finally, I would reconcile source and target counts and transaction amount totals, log the results, and generate a pass/fail report.

## Question 2

**Interviewer:** What would you do if source and target counts do not match?

**Strong answer:**

> I would first confirm that both counts are calculated using the same filters and batch date. Then I would compare primary keys to find records missing in target or extra in target. I would also check reject files, transformation logs, duplicate records, and late-arriving records. I would document the mismatch with counts and sample keys without exposing sensitive data.

## Question 3

**Interviewer:** How do you handle large files in Python?

**Strong answer:**

> If the file is too large for memory, I would process it in chunks using pandas `chunksize` or stream it line by line. If the data is distributed or very large, I would use PySpark or SQL/Hive/Spark to perform validations closer to the data.

## Question 4

**Interviewer:** How would you make your Python QA scripts reusable?

**Strong answer:**

> I would avoid hardcoding file names, paths, and rules. I would use `argparse` for command-line parameters, config files for validation rules, reusable functions for checks, logging for auditability, and a standard report format. I would also add unit tests for important validation functions.

## Question 5

**Interviewer:** What is the difference between unit testing and data validation testing?

**Strong answer:**

> Unit testing checks whether individual pieces of code behave correctly. Data validation testing checks whether actual data meets business, schema, and reconciliation rules. In a QA framework, I would use unit tests to test the validation functions, and then use those functions to validate real input datasets.

---

# 23. 10-Day Preparation Plan

## Day 1: Python basics

Practice:

- Variables
- Data types
- Lists
- Dictionaries
- Sets
- Functions
- Loops
- Conditions

Mini task:

```python
# Write a function that checks whether a value is in an allowed list.
```

## Day 2: File handling

Practice:

- `open()`
- `with` statement
- `pathlib`
- File exists check
- Empty file check
- Line count

Mini task:

```python
# Write a script that validates whether a file exists and is not empty.
```

## Day 3: CSV handling

Practice:

- `csv.reader`
- `csv.DictReader`
- Header validation
- Column count validation
- Pipe-delimited files

Mini task:

```python
# Validate that every row in a CSV has exactly 5 columns.
```

## Day 4: pandas basics

Practice:

- `read_csv`
- `head`
- `shape`
- `columns`
- `dtypes`
- `isnull`
- `duplicated`

Mini task:

```python
# Read a customer file and print duplicate customer IDs.
```

## Day 5: Data quality checks

Practice:

- Null checks
- Duplicate checks
- Allowed values
- Numeric conversion
- Date parsing
- Business rule checks

Mini task:

```python
# Validate account_status and balance fields.
```

## Day 6: Source-to-target reconciliation

Practice:

- Count comparison
- Key comparison
- Sum comparison
- Field-level comparison
- Hash comparison

Mini task:

```python
# Compare source.csv and target.csv using transaction_id.
```

## Day 7: Logging and argparse

Practice:

- `logging`
- `argparse`
- Exit codes
- Error handling

Mini task:

```python
# Convert a hardcoded validation script into a command-line tool.
```

## Day 8: pytest

Practice:

- Simple tests
- Assertions
- Parameterized tests
- Temporary files

Mini task:

```python
# Write unit tests for status validation and record count validation.
```

## Day 9: PySpark basics

Practice:

- SparkSession
- Read CSV
- Count records
- Null checks
- Duplicate checks
- GroupBy
- Source-target comparison

Mini task:

```python
# Write PySpark code to find duplicate transaction IDs.
```

## Day 10: Mock interview and mini project

Practice explaining:

- How to validate a data pipeline.
- How to debug count mismatches.
- How to handle large files.
- How to design a reusable QA framework.
- How to secure logs in a banking environment.

Mini task:

```python
# Complete the mini validation framework and explain it out loud.
```

---

# 24. Final Cheat Sheet

## Python basics

```python
list_values = [1, 2, 3]
unique_values = set(list_values)
record = {"id": 1, "status": "ACTIVE"}
```

## File checks

```python
from pathlib import Path

path = Path("customer.csv")
path.exists()
path.stat().st_size
```

## CSV module

```python
import csv

with open("file.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
```

## pandas

```python
import pandas as pd

df = pd.read_csv("file.csv")
df.shape
df.columns
df.dtypes
df.isnull().sum()
df.duplicated().sum()
```

## Duplicate check

```python
df[df.duplicated(subset=["customer_id"], keep=False)]
```

## Null check

```python
df[df["customer_id"].isnull()]
```

## Allowed values

```python
df[~df["status"].isin(["ACTIVE", "INACTIVE", "CLOSED"])]
```

## Date validation

```python
df["parsed_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
invalid_dates = df[df["parsed_date"].isnull()]
```

## Source-target reconciliation

```python
source_keys = set(source["id"])
target_keys = set(target["id"])

missing = source_keys - target_keys
extra = target_keys - source_keys
```

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Validation started")
logging.error("Validation failed")
```

## argparse

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()
```

## pytest

```python
def test_status():
    assert is_valid_status("ACTIVE") is True
```

## PySpark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder.appName("QA").getOrCreate()
df = spark.read.option("header", True).csv("/data/customer.csv")
df.count()
df.groupBy("customer_id").agg(count("*").alias("cnt")).filter(col("cnt") > 1).show()
```

---

# 25. Recommended Practice Datasets

You can create your own small datasets to practice. Use banking-style columns such as:

## Customer file

```csv
customer_id,customer_name,account_status,created_date
101,Alice,ACTIVE,2026-05-25
102,Bob,CLOSED,2026-05-25
103,,ACTIVE,2026-05-25
101,Alice,ACTIVE,2026-05-25
```

## Transaction file

```csv
transaction_id,account_id,transaction_amount,currency,trade_date
T001,A101,100.50,CAD,2026-05-25
T002,A102,-25.00,CAD,2026-05-25
T003,A103,ABC,USD,2026-05-25
T001,A101,100.50,CAD,2026-05-25
```

## Source-target comparison files

`source.csv`

```csv
id,amount,status
1,100,POSTED
2,200,POSTED
3,300,PENDING
```

`target.csv`

```csv
id,amount,status
1,100,POSTED
2,250,POSTED
4,400,POSTED
```

Practice answering:

- Which ID is missing in target?
- Which ID is extra in target?
- Which amount changed?
- What is the source total?
- What is the target total?
- Which records fail business rules?

---

# 26. References

These sources were used to align the guide with current public role requirements and official Python/data tooling documentation.

- RBC Careers — Sr. Quality Engineer posting mentioning Unix/Linux, Big Data/Hadoop, Python, and automated testing: https://jobs.rbc.com/ca/en/job/R-0000172211/Sr-Quality-Engineer
- RBC Careers — Senior Quality Engineer, Halifax posting mentioning Unix/Linux, SQL, and Python: https://jobs.rbc.com/ca/en/job/R-0000166737/Senior-Quality-Engineer-Halifax
- Python Standard Library documentation: https://docs.python.org/3/library/index.html
- Python `csv` module documentation: https://docs.python.org/3/library/csv.html
- Python `logging` HOWTO: https://docs.python.org/3/howto/logging.html
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- Python `unittest` documentation: https://docs.python.org/3/library/unittest.html
- pytest documentation: https://docs.pytest.org/en/stable/
- pandas `read_csv` documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- pandas I/O tools documentation: https://pandas.pydata.org/docs/user_guide/io.html
- PySpark documentation: https://spark.apache.org/docs/latest/api/python/index.html
- Apache Spark project site: https://spark.apache.org/

---

## Best Final Interview Summary

Use this as your closing answer when asked about Python for Big Data QA:

> I use Python to automate data quality checks such as file arrival validation, schema validation, record count reconciliation, duplicate detection, null checks, allowed-value checks, date and numeric validations, and source-to-target comparison. For smaller files, I use pandas; for large distributed datasets, I use PySpark or SQL. I make scripts production-ready with logging, command-line arguments, exception handling, clear exit codes, secure handling of sensitive data, and reusable validation reports.
