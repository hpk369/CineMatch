# QA Automation Interview Guide  
## For Big Data QA / Quality Engineer Role at RBC

**Target roles:** Big Data QA, Quality Engineer, Data QA Analyst, ETL QA Engineer, Automation QA Engineer, Data Validation Engineer  
**Target environment:** Banking / financial services / RBC-style enterprise data platforms  
**Main focus:** QA automation strategy, Python automation, pytest, Robot Framework, SQL validation, data pipeline testing, API testing, CI/CD, test reporting, and production-ready automation practices.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)  
2. [What QA Automation Means for Big Data QA](#2-what-qa-automation-means-for-big-data-qa)  
3. [QA Automation Skills Expected for This Role](#3-qa-automation-skills-expected-for-this-role)  
4. [Manual QA vs Automation QA vs Big Data QA Automation](#4-manual-qa-vs-automation-qa-vs-big-data-qa-automation)  
5. [Automation Testing Types](#5-automation-testing-types)  
6. [Core Automation Tools](#6-core-automation-tools)  
7. [Python for QA Automation](#7-python-for-qa-automation)  
8. [pytest for QA Automation](#8-pytest-for-qa-automation)  
9. [Robot Framework](#9-robot-framework)  
10. [SQL-Based Data Validation Automation](#10-sql-based-data-validation-automation)  
11. [Big Data QA Automation](#11-big-data-qa-automation)  
12. [ETL and Data Pipeline Automation](#12-etl-and-data-pipeline-automation)  
13. [API Testing Automation](#13-api-testing-automation)  
14. [File Validation Automation](#14-file-validation-automation)  
15. [Database Testing Automation](#15-database-testing-automation)  
16. [Test Data Management](#16-test-data-management)  
17. [Automation Framework Design](#17-automation-framework-design)  
18. [Logging and Reporting](#18-logging-and-reporting)  
19. [CI/CD Integration](#19-cicd-integration)  
20. [Automation in Banking Environments](#20-automation-in-banking-environments)  
21. [Common QA Automation Scenarios](#21-common-qa-automation-scenarios)  
22. [Common Automation Failures and Debugging](#22-common-automation-failures-and-debugging)  
23. [Interview Questions and Model Answers](#23-interview-questions-and-model-answers)  
24. [Hands-On Practice Tasks](#24-hands-on-practice-tasks)  
25. [10-Day Preparation Plan](#25-10-day-preparation-plan)  
26. [Final Cheat Sheet](#26-final-cheat-sheet)  
27. [Reference Links](#27-reference-links)

---

# 1. How to Use This Guide

This guide is designed for interview preparation for a **Big Data QA / Quality Engineer** role where QA automation is used to validate data pipelines, databases, APIs, files, and batch jobs.

For this type of role, QA automation is usually not just Selenium UI testing. It is more focused on:

- Automating SQL validation
- Automating source-to-target checks
- Automating file validation
- Automating data pipeline checks
- Automating API tests
- Automating audit/reject table validation
- Running validations through CI/CD pipelines
- Generating clear pass/fail reports
- Making repeatable checks for large-scale data

The most important mindset:

> In Big Data QA automation, the goal is to convert repeatable manual validation checks into reliable, reusable, maintainable automated tests that can run consistently across environments and data batches.

---

# 2. What QA Automation Means for Big Data QA

QA automation means using scripts, tools, and frameworks to run tests automatically.

In a Big Data QA role, automation usually checks:

- Did the source file arrive?
- Is the file not empty?
- Did the table load successfully?
- Do source and target counts match?
- Are there duplicate records?
- Are mandatory fields null?
- Are business rules applied correctly?
- Did invalid records go to reject tables?
- Did the audit table record correct counts?
- Did the pipeline stop when validation failed?
- Did the API return expected responses?
- Did the job logs show errors?

---

## Example manual vs automated validation

Manual:

```text
1. Log into database.
2. Run source count query.
3. Run target count query.
4. Copy results into Excel.
5. Compare counts manually.
6. Mark pass/fail.
```

Automated:

```text
1. Python script connects to source and target.
2. Script executes both count queries.
3. Script compares results.
4. Script writes pass/fail result.
5. CI/CD or Airflow fails the task if validation fails.
```

---

## Strong interview explanation

> For Big Data QA automation, I focus on automating repeatable data validation checks such as source-to-target counts, duplicate checks, null checks, business rule validation, audit table checks, reject record validation, and file checks. I usually use Python, SQL, pytest or Robot Framework, shell scripts, and CI/CD tools to make these validations repeatable and reportable.

---

# 3. QA Automation Skills Expected for This Role

For an RBC-style Big Data QA / Quality Engineer role, prioritize these skills:

| Category | Skills / Tools |
|---|---|
| Programming | Python, basic Java helpful |
| Test framework | pytest, Robot Framework, unittest |
| SQL automation | SQL queries, database connectors, source-to-target validation |
| Big Data | Hadoop, HDFS, Hive, Spark, PySpark |
| File validation | CSV, JSON, Parquet, ORC, control files |
| API testing | REST API testing, requests library, status codes, JSON validation |
| CI/CD | Jenkins, GitHub Actions, GitLab CI, Azure DevOps |
| Scheduling | Airflow, cron, enterprise schedulers |
| Reporting | HTML reports, JUnit XML, Allure, logs |
| Version control | Git |
| Data quality | Nulls, duplicates, schema, counts, reconciliation |
| Banking mindset | Auditability, traceability, data privacy, repeatability |

---

## Must-know automation stack

```text
Python
pytest
Robot Framework
SQL
Unix/Linux
Shell scripting
Git
Jenkins or another CI/CD tool
HDFS/Hive/Spark basics
API testing basics
```

---

## Nice-to-have stack

```text
Allure Reports
Great Expectations
Selenium
Postman/Newman
Docker
Kubernetes/OpenShift
Airflow
dbt tests
Kafka testing concepts
Cloud testing tools
```

---

# 4. Manual QA vs Automation QA vs Big Data QA Automation

## Manual QA

Manual QA means a tester executes test cases manually.

Example:

- Open application.
- Click buttons.
- Check output.
- Run SQL query manually.
- Compare results manually.

---

## Automation QA

Automation QA means test execution is automated.

Example:

- Selenium UI tests
- API tests
- Regression test scripts
- Automated test reports

---

## Big Data QA Automation

Big Data QA automation focuses on automating large-scale data checks.

Example:

- SQL count reconciliation
- PySpark duplicate checks
- HDFS file arrival checks
- Hive partition validation
- ETL audit table validation
- Reject record validation
- Pipeline validation tasks in Airflow

---

## Comparison table

| Area | Manual QA | General Automation QA | Big Data QA Automation |
|---|---|---|---|
| Main focus | Manual verification | Automated functional tests | Automated data validation |
| Common tools | Excel, SQL client | Selenium, API tools | Python, SQL, PySpark, Hive |
| Data size | Small to medium | Usually app-level | Large-scale data |
| Output | Test evidence | Test report | Reconciliation report |
| Key skill | Test thinking | Framework coding | Data + automation thinking |

---

# 5. Automation Testing Types

## Functional automation

Checks whether system functionality works as expected.

Example:

```text
API returns account details for valid account ID.
```

---

## Regression automation

Checks whether existing functionality still works after changes.

Example:

```text
Run daily source-to-target checks after each pipeline release.
```

---

## Smoke automation

Quick checks to confirm a build or deployment is basically working.

Example:

```text
DAG imports successfully.
Database connection works.
Target table exists.
```

---

## Data validation automation

Checks data quality and correctness.

Examples:

- Count match
- Sum match
- Duplicate check
- Null check
- Invalid value check
- Schema check

---

## API automation

Checks API responses.

Examples:

- Status code is 200
- Response schema is valid
- Required fields exist
- Error handling works

---

## Pipeline automation

Checks pipeline execution.

Examples:

- DAG completed successfully
- All tasks completed
- Audit counts match
- Reject records are correct
- Downstream task did not run after validation failure

---

# 6. Core Automation Tools

## Python

Used for:

- Test scripts
- Data validation
- File validation
- API testing
- Database query execution
- Report generation
- PySpark validation

---

## pytest

Used for:

- Python test automation
- Assertions
- Fixtures
- Parameterized tests
- Test reports
- CI/CD integration

---

## Robot Framework

Used for:

- Keyword-driven automation
- Acceptance testing
- API testing
- Database testing with libraries
- Easy-to-read test cases

---

## SQL

Used for:

- Count validation
- Null checks
- Duplicate checks
- Reconciliation
- Business rule checks
- Audit/reject validation

---

## Shell scripting

Used for:

- File checks
- HDFS commands
- Log scanning
- Job execution
- Simple automation wrappers

---

## CI/CD tools

Used for:

- Running automated tests on schedule or code changes
- Publishing reports
- Failing builds when validation fails

Examples:

- Jenkins
- GitHub Actions
- GitLab CI
- Azure DevOps

---

# 7. Python for QA Automation

Python is one of the most important tools for Big Data QA automation.

---

## Why Python is useful

Python can:

- Read files
- Connect to databases
- Run SQL queries
- Compare source and target results
- Call APIs
- Parse JSON
- Generate reports
- Run PySpark jobs
- Integrate with pytest and CI/CD

---

## Basic validation script

```python
def validate_count(source_count, target_count):
    if source_count == target_count:
        return "PASS"
    return "FAIL"


result = validate_count(1000, 1000)
print(result)
```

---

## Using assert

```python
source_count = 1000
target_count = 1000

assert source_count == target_count, "Source and target counts do not match"
```

If the assertion fails, the test fails.

---

## File exists validation

```python
from pathlib import Path

file_path = Path("customer_data.csv")

assert file_path.exists(), "File does not exist"
assert file_path.stat().st_size > 0, "File is empty"
```

---

## CSV record count validation

```python
import csv

def count_csv_records(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)
        return len(rows) - 1  # excluding header


actual_count = count_csv_records("customer_data.csv")
expected_count = 1000

assert actual_count == expected_count, f"Expected {expected_count}, got {actual_count}"
```

---

## JSON validation example

```python
import json

with open("api_response.json", "r", encoding="utf-8") as file:
    data = json.load(file)

assert "customer_id" in data
assert "account_status" in data
assert data["account_status"] in ["ACTIVE", "INACTIVE", "CLOSED"]
```

---

## Database query function

Example using a generic DB-API style connection:

```python
def run_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result
```

In real work, the exact connector depends on the database:

- `pyodbc`
- `psycopg2`
- `oracledb`
- `pymysql`
- `snowflake-connector-python`
- Spark SQL
- Hive connector

---

# 8. pytest for QA Automation

pytest is a popular Python testing framework.

It is useful for Big Data QA because it can run automated validation checks and integrate with CI/CD.

---

## Basic pytest test

File:

```text
test_counts.py
```

Code:

```python
def test_source_target_count():
    source_count = 1000
    target_count = 1000

    assert source_count == target_count
```

Run:

```bash
pytest test_counts.py
```

---

## Test with failure message

```python
def test_source_target_count():
    source_count = 1000
    target_count = 995

    assert source_count == target_count, (
        f"Count mismatch: source={source_count}, target={target_count}"
    )
```

---

## pytest fixtures

Fixtures provide reusable setup.

```python
import pytest

@pytest.fixture
def batch_id():
    return "BATCH_20260525"


def test_batch_id(batch_id):
    assert batch_id.startswith("BATCH_")
```

---

## Parameterized tests

Useful for running the same validation across multiple tables.

```python
import pytest

@pytest.mark.parametrize(
    "table_name, key_column",
    [
        ("customer_dim", "customer_id"),
        ("transaction_fact", "transaction_id"),
        ("account_dim", "account_id"),
    ],
)
def test_primary_key_not_null(table_name, key_column):
    query = f"""
        SELECT COUNT(*)
        FROM {table_name}
        WHERE {key_column} IS NULL
    """

    # result = run_query(query)
    result = 0

    assert result == 0, f"{table_name}.{key_column} has null values"
```

---

## pytest for data quality checks

```python
def test_no_duplicate_transaction_ids(db_connection):
    query = """
        SELECT COUNT(*)
        FROM (
            SELECT transaction_id
            FROM transaction_fact
            WHERE business_date = '2026-05-25'
            GROUP BY transaction_id
            HAVING COUNT(*) > 1
        ) d
    """

    result = run_scalar_query(db_connection, query)

    assert result == 0, f"Found {result} duplicate transaction IDs"
```

---

## pytest command examples

```bash
pytest
pytest -v
pytest tests/test_counts.py
pytest -k "count"
pytest --junitxml=reports/results.xml
```

---

## pytest with CI/CD

JUnit XML output is useful for CI/CD:

```bash
pytest --junitxml=reports/qa_results.xml
```

Jenkins, GitLab CI, and Azure DevOps can read test result files and display pass/fail results.

---

# 9. Robot Framework

Robot Framework is a keyword-driven automation framework.

It is commonly used in enterprise QA environments because test cases are readable and structured.

---

## Robot Framework file structure

A Robot Framework test file usually has sections:

```robot
*** Settings ***
Library    OperatingSystem

*** Variables ***
${FILE_PATH}    customer_data.csv

*** Test Cases ***
Validate File Exists
    File Should Exist    ${FILE_PATH}
```

---

## Basic test case

```robot
*** Settings ***
Library    OperatingSystem

*** Test Cases ***
Validate Input File Exists
    File Should Exist    customer_data.csv
```

---

## Variable example

```robot
*** Variables ***
${EXPECTED_COUNT}    1000
${ACTUAL_COUNT}      1000

*** Test Cases ***
Validate Counts Match
    Should Be Equal As Integers    ${ACTUAL_COUNT}    ${EXPECTED_COUNT}
```

---

## Keyword example

```robot
*** Keywords ***
Validate Counts Match
    [Arguments]    ${source_count}    ${target_count}
    Should Be Equal As Integers    ${source_count}    ${target_count}

*** Test Cases ***
Validate Source Target Count
    Validate Counts Match    1000    1000
```

---

## Robot Framework for API testing

Using RequestsLibrary:

```robot
*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Validate Customer API
    Create Session    customer_api    https://api.example.com
    ${response}=    GET On Session    customer_api    /customers/123
    Should Be Equal As Integers    ${response.status_code}    200
```

---

## Robot Framework for database testing

Using a database library, a test can execute SQL and validate results.

Example concept:

```robot
*** Test Cases ***
Validate No Duplicate Transactions
    ${result}=    Query    SELECT COUNT(*) FROM duplicate_check_result
    Should Be Equal As Integers    ${result[0][0]}    0
```

Exact syntax depends on the database library used.

---

## When to mention Robot Framework

Use this interview answer:

> Robot Framework is useful when teams want readable keyword-driven tests. I can create reusable keywords for database queries, file checks, API calls, and validation logic. For Big Data QA, Robot Framework can be used to automate source-to-target checks, count validations, API tests, and pipeline smoke tests.

---

# 10. SQL-Based Data Validation Automation

SQL-based validation is one of the most important areas for Big Data QA automation.

---

## Common SQL checks to automate

| Check | SQL Pattern |
|---|---|
| Count check | `COUNT(*)` |
| Duplicate check | `GROUP BY ... HAVING COUNT(*) > 1` |
| Null check | `WHERE column IS NULL` |
| Domain check | `WHERE value NOT IN (...)` |
| Sum check | `SUM(amount)` |
| Missing records | `LEFT JOIN ... WHERE target.key IS NULL` |
| Field mismatch | Join source and target, compare fields |
| Audit check | Source = target + reject |
| Partition check | Filter by business date/batch |

---

## Count validation SQL

```sql
SELECT COUNT(*) AS source_count
FROM source_transaction
WHERE business_date = '2026-05-25';

SELECT COUNT(*) AS target_count
FROM target_transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Automated count comparison in Python

```python
def test_source_target_count(db_connection):
    source_query = """
        SELECT COUNT(*)
        FROM source_transaction
        WHERE business_date = '2026-05-25'
    """

    target_query = """
        SELECT COUNT(*)
        FROM target_transaction_fact
        WHERE business_date = '2026-05-25'
    """

    source_count = run_scalar_query(db_connection, source_query)
    target_count = run_scalar_query(db_connection, target_query)

    assert source_count == target_count, (
        f"Count mismatch: source={source_count}, target={target_count}"
    )
```

---

## Duplicate validation SQL

```sql
SELECT transaction_id, COUNT(*) AS duplicate_count
FROM target_transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Automated duplicate check

```python
def test_no_duplicate_transactions(db_connection):
    query = """
        SELECT COUNT(*)
        FROM (
            SELECT transaction_id
            FROM target_transaction_fact
            WHERE business_date = '2026-05-25'
            GROUP BY transaction_id
            HAVING COUNT(*) > 1
        ) d
    """

    duplicate_count = run_scalar_query(db_connection, query)

    assert duplicate_count == 0, f"Found {duplicate_count} duplicate transaction IDs"
```

---

## Null validation SQL

```sql
SELECT COUNT(*)
FROM target_transaction_fact
WHERE business_date = '2026-05-25'
  AND customer_id IS NULL;
```

---

## Domain validation SQL

```sql
SELECT COUNT(*)
FROM target_transaction_fact
WHERE business_date = '2026-05-25'
  AND currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP');
```

---

# 11. Big Data QA Automation

Big Data QA automation involves validating data stored and processed by tools such as:

- HDFS
- Hive
- Spark
- PySpark
- Hadoop jobs
- Airflow DAGs
- Kafka pipelines
- Cloud data lakes

---

## HDFS file validation automation

Shell command:

```bash
hdfs dfs -test -e /landing/transactions/txn_20260525.csv
```

Python wrapper:

```python
import subprocess

def hdfs_path_exists(path):
    result = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def test_hdfs_file_exists():
    path = "/landing/transactions/txn_20260525.csv"
    assert hdfs_path_exists(path), f"HDFS file missing: {path}"
```

---

## HDFS record count automation

```python
import subprocess

def get_hdfs_line_count(path):
    command = f"hdfs dfs -cat {path} | wc -l"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return int(result.stdout.strip())


def test_hdfs_file_record_count():
    path = "/landing/transactions/txn_20260525.csv"
    actual_count = get_hdfs_line_count(path)
    expected_count = 100000

    assert actual_count == expected_count
```

---

## Hive validation automation

Use SQL queries to validate Hive tables:

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

In automation, use Python or Spark to execute this query and compare the result.

---

## PySpark validation automation

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("QAValidation") \
    .enableHiveSupport() \
    .getOrCreate()

df = spark.table("finance.transaction_fact") \
    .filter(col("business_date") == "2026-05-25")

assert df.count() > 0
```

---

## PySpark duplicate check

```python
from pyspark.sql.functions import col

duplicates = df.groupBy("transaction_id") \
    .count() \
    .filter(col("count") > 1)

assert duplicates.count() == 0, "Duplicate transaction IDs found"
```

---

## PySpark null check

```python
null_count = df.filter(col("customer_id").isNull()).count()

assert null_count == 0, f"Found {null_count} null customer IDs"
```

---

## PySpark source-to-target count check

```python
source_df = spark.table("finance.source_transaction") \
    .filter(col("business_date") == "2026-05-25")

target_df = spark.table("finance.target_transaction_fact") \
    .filter(col("business_date") == "2026-05-25")

source_count = source_df.count()
target_count = target_df.count()

assert source_count == target_count, (
    f"Count mismatch: source={source_count}, target={target_count}"
)
```

---

# 12. ETL and Data Pipeline Automation

ETL pipeline automation validates the data flow from source to target.

---

## ETL automation checkpoints

| Stage | Automated Check |
|---|---|
| Source | Source file/table available |
| Landing | File exists, file count, file size |
| Raw | Schema, record count |
| Staging | Nulls, duplicates, invalid values |
| Transform | Business rules, joins, derivations |
| Target | Count, sum, field-level comparison |
| Reject | Invalid records captured |
| Audit | Counts and status recorded |
| Downstream | Dependent tasks triggered correctly |

---

## Example pipeline validation flow

```text
test_file_arrival
    ↓
test_raw_count
    ↓
test_target_count
    ↓
test_duplicate_keys
    ↓
test_null_mandatory_fields
    ↓
test_audit_reconciliation
```

---

## Airflow integration idea

An Airflow DAG can run validation scripts as tasks:

```python
validate_counts = BashOperator(
    task_id="validate_counts",
    bash_command="pytest tests/test_counts.py --junitxml=reports/counts.xml"
)
```

If the pytest script fails, the Airflow task should fail.

---

## Important automation rule

Validation scripts must return a failing exit code when validation fails.

Bad:

```python
if source_count != target_count:
    print("FAIL")
```

This only prints failure.

Good:

```python
if source_count != target_count:
    raise AssertionError("Source and target counts do not match")
```

This fails the automation task.

---

# 13. API Testing Automation

API testing may be part of QA automation even for data roles.

APIs may be used to:

- Trigger a job
- Check job status
- Retrieve reference data
- Validate data services
- Check customer/account information

---

## API testing with Python requests

```python
import requests

def test_customer_api_status():
    response = requests.get("https://api.example.com/customers/123")

    assert response.status_code == 200
```

---

## Validate JSON response

```python
def test_customer_api_response_schema():
    response = requests.get("https://api.example.com/customers/123")
    data = response.json()

    assert "customer_id" in data
    assert "status" in data
    assert data["status"] in ["ACTIVE", "INACTIVE", "CLOSED"]
```

---

## Validate error response

```python
def test_customer_api_invalid_id():
    response = requests.get("https://api.example.com/customers/invalid-id")

    assert response.status_code in [400, 404]
```

---

## Common API validation points

| Check | Example |
|---|---|
| Status code | 200, 201, 400, 401, 404, 500 |
| Response time | API responds within expected time |
| Required fields | JSON contains required fields |
| Data type | Amount is numeric |
| Business rule | Status is valid |
| Error handling | Invalid request returns proper error |
| Authentication | Unauthorized request is rejected |

---

# 14. File Validation Automation

File validation is a major part of Big Data QA automation.

---

## Common file checks

| Check | Example |
|---|---|
| File exists | File path present |
| File is not empty | Size > 0 |
| File count | Expected number of files |
| Record count | Expected number of rows |
| Header check | Header matches expected |
| Trailer check | Trailer count matches records |
| Delimiter check | Correct delimiter |
| Schema check | Expected columns |
| Duplicate check | Duplicate keys |
| Null check | Mandatory fields not blank |
| Naming convention | Date and source in filename |

---

## Local file validation

```python
from pathlib import Path

def test_file_exists():
    path = Path("customer_20260525.csv")

    assert path.exists(), "Input file does not exist"
    assert path.stat().st_size > 0, "Input file is empty"
```

---

## Header validation

```python
def test_csv_header():
    expected_header = "customer_id,first_name,last_name,email,status"

    with open("customer_20260525.csv", "r", encoding="utf-8") as file:
        actual_header = file.readline().strip()

    assert actual_header == expected_header
```

---

## Delimiter validation

```python
def test_csv_column_count():
    expected_columns = 5

    with open("customer_20260525.csv", "r", encoding="utf-8") as file:
        header = file.readline()

        for line_number, line in enumerate(file, start=2):
            columns = line.strip().split(",")
            assert len(columns) == expected_columns, (
                f"Invalid column count at line {line_number}"
            )
```

---

## Trailer validation example

Suppose file trailer is:

```text
TRAILER|1000
```

Validation:

```python
def test_trailer_count():
    with open("customer_20260525.txt", "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]

    trailer = lines[-1]
    trailer_count = int(trailer.split("|")[1])
    data_count = len(lines) - 2  # excluding header and trailer

    assert trailer_count == data_count, (
        f"Trailer count={trailer_count}, actual data count={data_count}"
    )
```

---

# 15. Database Testing Automation

Database testing automation validates data stored in tables.

---

## Common database checks

| Check | SQL Example |
|---|---|
| Table exists | Metadata query |
| Count check | `COUNT(*)` |
| Duplicate check | `GROUP BY HAVING` |
| Null check | `IS NULL` |
| Domain check | `NOT IN` |
| Referential integrity | Join with reference table |
| Aggregation check | `SUM`, `MIN`, `MAX` |
| Source-target comparison | Joins / EXCEPT |
| Audit check | Count reconciliation |
| SCD check | One current record |

---

## Generic database test structure

```python
def test_database_validation(db_connection):
    query = "SELECT COUNT(*) FROM transaction_fact"
    count = run_scalar_query(db_connection, query)

    assert count > 0
```

---

## Source-to-target comparison

```python
def test_source_target_amount_sum(db_connection):
    source_sum_query = """
        SELECT SUM(amount)
        FROM source_transaction
        WHERE business_date = '2026-05-25'
    """

    target_sum_query = """
        SELECT SUM(amount)
        FROM target_transaction_fact
        WHERE business_date = '2026-05-25'
    """

    source_sum = run_scalar_query(db_connection, source_sum_query)
    target_sum = run_scalar_query(db_connection, target_sum_query)

    assert source_sum == target_sum, (
        f"Amount mismatch: source={source_sum}, target={target_sum}"
    )
```

---

## Audit table validation

```python
def test_audit_reconciliation(db_connection):
    query = """
        SELECT COUNT(*)
        FROM etl_audit
        WHERE batch_id = 'BATCH_20260525'
          AND source_count <> target_count + reject_count
    """

    failed_jobs = run_scalar_query(db_connection, query)

    assert failed_jobs == 0, f"Audit reconciliation failed for {failed_jobs} jobs"
```

---

# 16. Test Data Management

Test data management means preparing and controlling the data used for testing.

---

## Why test data matters

Automation is only reliable if test data is reliable.

Bad test data can cause:

- False failures
- False passes
- Unstable tests
- Hard-to-debug results
- Privacy/security issues

---

## Types of test data

| Type | Example |
|---|---|
| Positive data | Valid customer record |
| Negative data | Invalid currency code |
| Boundary data | Amount = 0, max amount |
| Duplicate data | Same transaction ID twice |
| Null data | Missing customer ID |
| Historical data | Old transaction date |
| Incremental data | Updated record |
| Rejected data | Record expected to fail validation |

---

## Banking test data considerations

In banking environments:

- Do not expose real customer data.
- Use masked or synthetic data where possible.
- Avoid printing sensitive values in logs.
- Store test data securely.
- Follow environment access rules.
- Keep auditability of test execution.

---

## Test data example

```csv
transaction_id,account_id,customer_id,amount,currency_code,transaction_type
T001,A001,C001,100.00,CAD,PAYMENT
T002,A002,C002,-50.00,CAD,REVERSAL
T003,A003,,200.00,USD,PAYMENT
T004,A004,C004,500.00,XYZ,PAYMENT
```

Validation expectations:

| Transaction | Expected Result |
|---|---|
| T001 | Valid |
| T002 | Valid if reversal allows negative |
| T003 | Reject due to missing customer_id |
| T004 | Reject due to invalid currency |

---

# 17. Automation Framework Design

A good automation framework is reusable, maintainable, and easy to run.

---

## Basic framework structure

```text
qa_automation/
│
├── configs/
│   ├── dev.yaml
│   ├── qa.yaml
│   └── prod.yaml
│
├── tests/
│   ├── test_file_checks.py
│   ├── test_count_reconciliation.py
│   ├── test_duplicate_checks.py
│   ├── test_null_checks.py
│   └── test_audit_checks.py
│
├── utils/
│   ├── db_utils.py
│   ├── file_utils.py
│   ├── hdfs_utils.py
│   ├── spark_utils.py
│   └── report_utils.py
│
├── sql/
│   ├── source_count.sql
│   ├── target_count.sql
│   ├── duplicate_check.sql
│   └── audit_check.sql
│
├── reports/
│
├── requirements.txt
│
└── README.md
```

---

## Key design principles

| Principle | Meaning |
|---|---|
| Reusable | Common functions shared |
| Config-driven | Environment values in config files |
| Data-driven | Tests can run for many tables |
| Clear assertions | Failures explain the issue |
| Logging | Execution details captured |
| Reporting | Results are easy to review |
| Secure | No passwords in code |
| CI/CD-ready | Can run from pipeline |

---

## Config example

```yaml
environment: qa
business_date: "2026-05-25"

database:
  host: "qa-db.example.com"
  name: "finance"
  user: "qa_user"

tables:
  source_transaction: "source_transaction"
  target_transaction: "target_transaction_fact"
```

---

## Data-driven validation config

```yaml
validations:
  - name: transaction_count_check
    source_table: source_transaction
    target_table: target_transaction_fact
    filter_column: business_date
    filter_value: "2026-05-25"

  - name: customer_duplicate_check
    table: customer_dim
    key_column: customer_id
```

---

## Data-driven pytest idea

```python
import pytest

validations = [
    ("source_transaction", "target_transaction_fact", "business_date", "2026-05-25"),
    ("source_customer", "target_customer_dim", "business_date", "2026-05-25"),
]

@pytest.mark.parametrize(
    "source_table,target_table,filter_column,filter_value",
    validations
)
def test_count_reconciliation(source_table, target_table, filter_column, filter_value):
    source_count = get_count(source_table, filter_column, filter_value)
    target_count = get_count(target_table, filter_column, filter_value)

    assert source_count == target_count
```

---

# 18. Logging and Reporting

Good automation must produce clear logs and reports.

---

## What to log

Log:

- Test name
- Environment
- Batch ID or business date
- Source table/path
- Target table/path
- Query name
- Expected value
- Actual value
- Pass/fail result
- Error message
- Timestamp

Do not log sensitive data such as:

- Full account numbers
- Card numbers
- Passwords
- Customer personal information
- Authentication tokens

---

## Python logging example

```python
import logging

logging.basicConfig(
    filename="qa_automation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting count validation")

source_count = 1000
target_count = 995

if source_count != target_count:
    logging.error(
        "Count mismatch: source=%s, target=%s",
        source_count,
        target_count
    )
```

---

## pytest report examples

Run with verbose output:

```bash
pytest -v
```

Generate JUnit XML:

```bash
pytest --junitxml=reports/results.xml
```

Generate HTML report with plugin:

```bash
pytest --html=reports/report.html
```

---

## Robot Framework reports

Robot Framework automatically creates:

```text
output.xml
log.html
report.html
```

Run:

```bash
robot tests/
```

---

## What a good QA report contains

| Field | Example |
|---|---|
| Test name | Source Target Count Check |
| Environment | QA |
| Batch ID | BATCH_20260525 |
| Status | PASS / FAIL |
| Expected | 1,000,000 |
| Actual | 999,950 |
| Difference | 50 |
| Failure reason | Target missing 50 records |
| Evidence | Query name, log file, screenshot if applicable |

---

# 19. CI/CD Integration

CI/CD integration makes automation useful for teams.

---

## Why CI/CD matters

Without CI/CD, tests may run only when someone remembers.

With CI/CD:

- Tests run automatically after code changes.
- Tests run on schedules.
- Reports are published.
- Failed tests can block deployment.
- Teams get faster feedback.

---

## Jenkins pipeline example

```groovy
pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run QA Automation') {
            steps {
                sh 'pytest tests/ --junitxml=reports/results.xml'
            }
        }
    }

    post {
        always {
            junit 'reports/results.xml'
        }
    }
}
```

---

## GitHub Actions example

```yaml
name: QA Automation

on:
  push:
  workflow_dispatch:

jobs:
  run-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ --junitxml=reports/results.xml
```

---

## CI/CD validation examples

Pipeline can run:

```bash
pytest tests/test_smoke.py
pytest tests/test_data_quality.py
pytest tests/test_reconciliation.py
robot tests/
dbt test
```

---

## Strong interview answer

> I would integrate automation into CI/CD so validations run consistently after code changes or on a schedule. The automation should return proper exit codes, publish reports, and stop deployment or downstream tasks if critical validation fails.

---

# 20. Automation in Banking Environments

Banking environments require extra care.

---

## Key principles

| Principle | Meaning |
|---|---|
| Accuracy | Financial data must reconcile |
| Auditability | Test results should be traceable |
| Security | Sensitive data must be protected |
| Repeatability | Tests should run consistently |
| Access control | Only authorized users/tools access data |
| Evidence | Test results should be reviewable |
| Failure control | Downstream jobs should stop on critical failures |

---

## Sensitive data rules

Avoid logging:

- Account numbers
- Card numbers
- SIN/SSN
- Full customer names
- Addresses
- Authentication tokens
- Passwords
- Secrets

Better logging:

```text
FAIL: 25 records failed customer_id validation for batch BATCH_20260525
```

Avoid:

```text
FAIL: Customer John Smith, account 123456789 failed validation
```

---

## Banking-specific validations

Examples:

- Transaction amount reconciliation
- Currency code validation
- Account/customer reference validation
- Duplicate transaction detection
- Negative amount rule validation
- Trade date vs processing date validation
- Audit/reject validation
- Regulatory report totals

---

# 21. Common QA Automation Scenarios

## Scenario 1: Source and target count validation

Automated check:

```python
assert source_count == target_count
```

Interview explanation:

> I would automate source and target count comparison for each batch and fail the automation if counts do not match, unless reject counts explain the difference.

---

## Scenario 2: Source count equals target plus reject count

```python
assert source_count == target_count + reject_count
```

This is common in ETL pipelines where invalid records are rejected.

---

## Scenario 3: Duplicate validation

```sql
SELECT transaction_id, COUNT(*)
FROM target_transaction_fact
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

Automation expected result:

```text
0 duplicate keys
```

---

## Scenario 4: Null mandatory field validation

```sql
SELECT COUNT(*)
FROM target_transaction_fact
WHERE customer_id IS NULL;
```

Automation expected result:

```text
0 null customer IDs
```

---

## Scenario 5: Pipeline should stop on validation failure

Expected behavior:

```text
validate_target fails → publish_report should not run
```

QA checks:

- Task dependency
- Trigger rule
- Exit code
- Logs
- Alert generated

---

## Scenario 6: Smoke test after deployment

Smoke checks:

- DAG loads successfully
- Database connection works
- Target table exists
- One simple count query runs
- Required config exists
- Secrets are accessible through approved method

---

# 22. Common Automation Failures and Debugging

## Failure 1: Test fails due to environment issue

Examples:

- Database unavailable
- Network issue
- Missing credentials
- HDFS permission issue
- Wrong config file

Debug:

- Check environment config.
- Check connection.
- Check credentials.
- Check logs.
- Retry only if issue is temporary.

---

## Failure 2: Test fails due to real data issue

Examples:

- Count mismatch
- Duplicate records
- Null mandatory fields
- Invalid currency codes
- Missing target records

Debug:

- Confirm query.
- Check source/staging/target.
- Check reject table.
- Check audit table.
- Review job logs.
- Identify failing records.

---

## Failure 3: Flaky automation test

Flaky means the test sometimes passes and sometimes fails without code/data changes.

Causes:

- Timing issue
- Unstable test data
- Dependency not ready
- Parallel execution conflict
- Hardcoded dates
- Environment instability

Fix:

- Add proper waits/sensors.
- Use stable test data.
- Avoid hardcoded dates.
- Isolate tests.
- Make setup/teardown reliable.

---

## Failure 4: Test passed but data is wrong

Possible causes:

- Test only checked count.
- Test did not check business rules.
- Test did not compare field-level values.
- Test ignored reject records.
- Test data was not representative.

Lesson:

> Count checks are important, but they are not enough. Strong automation also checks data quality, sums, business rules, and field-level mappings.

---

## Failure 5: Secrets exposed in logs

This is serious.

Prevention:

- Use environment variables or secret managers.
- Never print passwords or tokens.
- Mask sensitive logs.
- Avoid logging full SQL results with sensitive data.

---

# 23. Interview Questions and Model Answers

## Q1. What is QA automation?

**Answer:**  
QA automation is the process of using scripts and tools to run repeatable tests automatically. In Big Data QA, this includes automating file checks, SQL validations, source-to-target reconciliation, duplicate checks, null checks, audit checks, and pipeline validations.

---

## Q2. What automation tools have you used or would you use?

**Answer:**  
For Big Data QA, I would use Python, pytest, Robot Framework, SQL scripts, shell scripts, PySpark, and CI/CD tools like Jenkins or Azure DevOps. For API testing, I would use Python requests or Robot Framework RequestsLibrary. For pipeline validation, I would integrate tests with Airflow or CI/CD.

---

## Q3. Why is Python useful in QA automation?

**Answer:**  
Python is useful because it can connect to databases, read files, call APIs, run validations, generate reports, and integrate with pytest, Robot Framework, PySpark, and CI/CD tools.

---

## Q4. What is pytest?

**Answer:**  
pytest is a Python testing framework. It supports assertions, fixtures, parameterization, test discovery, and report generation. It is useful for automating SQL validations, API tests, and data quality checks.

---

## Q5. What is Robot Framework?

**Answer:**  
Robot Framework is a keyword-driven automation framework. It allows readable test cases and reusable keywords. It can be used for API testing, database testing, file validation, and acceptance testing.

---

## Q6. How would you automate source-to-target validation?

**Answer:**  
I would create SQL queries for source count, target count, sum reconciliation, duplicate checks, and field-level comparisons. Then I would write Python or pytest tests to execute those queries, compare expected and actual results, and produce reports. The automation should fail if critical checks do not pass.

---

## Q7. How do you validate duplicates automatically?

**Answer:**  
I would run a grouped SQL query on the business key or primary key and check whether any key has count greater than one. In automation, the expected duplicate count should be zero.

---

## Q8. How do you make automation maintainable?

**Answer:**  
I would use reusable utility functions, config files, parameterized tests, clear naming, logging, reporting, and separate SQL files. I would avoid hardcoding environment values and keep credentials out of code.

---

## Q9. How do you handle failed automation tests?

**Answer:**  
First, I check if the failure is due to the automation itself, the environment, or a real data issue. Then I review logs, queries, test data, audit tables, and pipeline logs. If it is a real data issue, I document the failing records or counts and report it with evidence.

---

## Q10. What is the difference between assertion failure and script error?

**Answer:**  
An assertion failure means the test ran successfully but the expected condition was not met. A script error means the automation code itself failed due to issues like connection error, syntax error, missing file, or bad configuration.

---

## Q11. How do you integrate automation with CI/CD?

**Answer:**  
I would add test execution commands to the CI/CD pipeline, such as `pytest` or `robot`. The pipeline should install dependencies, run tests, publish reports, and fail the build or stage if critical tests fail.

---

## Q12. What is a smoke test in data pipeline automation?

**Answer:**  
A smoke test is a quick validation that confirms the basic system is working. For a data pipeline, smoke tests can check database connectivity, table existence, DAG import, source file availability, and a simple count query.

---

## Q13. What should you not automate?

**Answer:**  
I would avoid automating unstable or one-time checks, tests with unclear expected results, and tests that require heavy manual interpretation. Automation is best for repeatable, deterministic checks.

---

## Q14. What is a good automation framework?

**Answer:**  
A good automation framework is reusable, configurable, maintainable, secure, and CI/CD-ready. It should have clear logs, reports, reusable utilities, environment configs, data-driven tests, and meaningful failure messages.

---

## Q15. How do you ensure automation does not expose sensitive data?

**Answer:**  
I avoid logging sensitive fields, use masked or synthetic test data, store secrets securely, avoid hardcoded credentials, and log only counts, IDs when approved, and summarized failure information.

---

# 24. Hands-On Practice Tasks

## Task 1: Create a pytest count validation

Write a pytest test that compares source and target counts.

Expected idea:

```python
def test_source_target_count():
    source_count = 1000
    target_count = 1000

    assert source_count == target_count
```

---

## Task 2: Create a duplicate check query

```sql
SELECT transaction_id, COUNT(*)
FROM target_transaction_fact
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Task 3: Create a Robot Framework file check

```robot
*** Settings ***
Library    OperatingSystem

*** Test Cases ***
Validate Input File Exists
    File Should Exist    customer_data.csv
```

---

## Task 4: Create an API test with Python

```python
import requests

def test_api_status():
    response = requests.get("https://api.example.com/health")
    assert response.status_code == 200
```

---

## Task 5: Create a null check SQL

```sql
SELECT COUNT(*)
FROM customer_dim
WHERE customer_id IS NULL;
```

---

## Task 6: Create an audit reconciliation test

```python
def test_audit_reconciliation():
    source_count = 1000
    target_count = 980
    reject_count = 20

    assert source_count == target_count + reject_count
```

---

## Task 7: Create a PySpark duplicate check

```python
duplicates = df.groupBy("transaction_id").count().filter("count > 1")

assert duplicates.count() == 0
```

---

## Task 8: Create a Jenkins command to run tests

```bash
pytest tests/ --junitxml=reports/results.xml
```

---

# 25. 10-Day Preparation Plan

## Day 1: QA automation fundamentals

Study:

- What is automation?
- Manual vs automation testing
- What should be automated?
- Data QA automation examples
- Basic test strategy

Goal: Explain QA automation clearly.

---

## Day 2: Python automation basics

Practice:

- Functions
- Assertions
- File reading
- JSON parsing
- Exception handling
- Logging

Goal: Write basic validation scripts.

---

## Day 3: pytest basics

Practice:

- Test functions
- Assertions
- Fixtures
- Parameterized tests
- Running pytest
- Generating JUnit XML

Goal: Build small automated test suites.

---

## Day 4: Robot Framework basics

Practice:

- Test cases
- Keywords
- Variables
- Running robot tests
- Reading reports

Goal: Explain keyword-driven testing.

---

## Day 5: SQL validation automation

Practice:

- Count checks
- Null checks
- Duplicate checks
- Sum checks
- Source-to-target joins
- Audit reconciliation

Goal: Automate database checks.

---

## Day 6: File and HDFS validation

Practice:

- File exists checks
- File size checks
- Header validation
- Record count validation
- HDFS command wrappers

Goal: Automate file-level checks.

---

## Day 7: API testing automation

Practice:

- GET request
- POST request
- Status code check
- JSON field validation
- Error response validation

Goal: Automate basic REST API checks.

---

## Day 8: Big Data automation

Practice:

- Hive count checks
- PySpark null checks
- PySpark duplicate checks
- Source-to-target count comparison
- Partition validation

Goal: Apply automation to large-scale data.

---

## Day 9: CI/CD and reporting

Practice:

- pytest reports
- Robot reports
- Jenkins test command
- JUnit XML
- Logging
- Failure messages

Goal: Make tests pipeline-ready.

---

## Day 10: Mock interview

Practice answering:

- What is QA automation?
- What tools would you use?
- How do you automate source-to-target validation?
- How do you use pytest?
- How do you use Robot Framework?
- How do you debug automation failures?
- How do you integrate tests into CI/CD?
- How do you protect sensitive data in automation?

Goal: Sound practical and confident.

---

# 26. Final Cheat Sheet

## Must-know tools

```text
Python
pytest
Robot Framework
SQL
Shell scripting
Git
Jenkins / CI/CD
PySpark
Hive
HDFS
API testing tools
```

---

## Most important automation checks

```text
File exists
File is not empty
Record count
Source-to-target count
Source-to-target sum
Duplicate keys
Null mandatory fields
Invalid domain values
Business rules
Audit reconciliation
Reject validation
Partition validation
API response validation
Pipeline status validation
```

---

## pytest essentials

```python
def test_example():
    expected = 100
    actual = 100

    assert actual == expected
```

Run:

```bash
pytest -v
pytest --junitxml=reports/results.xml
```

---

## Robot Framework essentials

```robot
*** Settings ***
Library    OperatingSystem

*** Test Cases ***
Validate File Exists
    File Should Exist    customer_data.csv
```

Run:

```bash
robot tests/
```

---

## SQL duplicate check

```sql
SELECT business_key, COUNT(*)
FROM target_table
GROUP BY business_key
HAVING COUNT(*) > 1;
```

---

## SQL null check

```sql
SELECT COUNT(*)
FROM target_table
WHERE mandatory_column IS NULL;
```

---

## SQL audit check

```sql
SELECT
    source_count,
    target_count,
    reject_count,
    CASE
        WHEN source_count = target_count + reject_count THEN 'PASS'
        ELSE 'FAIL'
    END AS validation_status
FROM etl_audit
WHERE batch_id = 'BATCH_20260525';
```

---

## Strong interview answer

If asked, **“How would you approach QA automation for a Big Data pipeline?”**, say:

> I would identify repeatable validation checks and automate them using Python, SQL, pytest or Robot Framework. I would start with file arrival, schema, count, duplicate, null, and business rule checks. Then I would automate source-to-target reconciliation, audit table validation, reject table validation, and partition checks. I would integrate these tests into Airflow or CI/CD so failures stop downstream tasks and generate clear reports. In a banking environment, I would also make sure automation is auditable, repeatable, and does not expose sensitive data.

---

# 27. Reference Links

Use these for deeper study:

- Python Documentation: https://docs.python.org/3/
- pytest Documentation: https://docs.pytest.org/
- Robot Framework Documentation: https://robotframework.org/robotframework/
- Selenium Documentation: https://www.selenium.dev/documentation/
- Requests Library Documentation: https://requests.readthedocs.io/
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Apache Hive Documentation: https://hive.apache.org/docs/latest/
- Apache Hadoop FileSystem Shell: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html
- Jenkins Documentation: https://www.jenkins.io/doc/
- Allure Framework: https://allurereport.org/
