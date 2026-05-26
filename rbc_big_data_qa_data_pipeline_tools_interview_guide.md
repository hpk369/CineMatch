# Data Pipeline Tools Interview Guide  
## For Big Data QA / Quality Engineer Role at RBC

**Target roles:** Big Data QA, Data QA Analyst, ETL QA Engineer, Quality Engineer, Data Validation Engineer  
**Target environment:** Banking / financial services / RBC-style enterprise data platforms  
**Main focus:** Airflow, Oozie, Kafka, dbt, ETL/ELT tools, pipeline orchestration, scheduling, monitoring, audit/reject validation, CI/CD, and production debugging.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)  
2. [What “Data Pipeline Tools” Means](#2-what-data-pipeline-tools-means)  
3. [Why Pipeline Tools Matter in Big Data QA](#3-why-pipeline-tools-matter-in-big-data-qa)  
4. [End-to-End Data Pipeline Flow](#4-end-to-end-data-pipeline-flow)  
5. [Main Categories of Data Pipeline Tools](#5-main-categories-of-data-pipeline-tools)  
6. [Apache Airflow](#6-apache-airflow)  
7. [Airflow DAGs, Tasks, Operators, Sensors, and Scheduling](#7-airflow-dags-tasks-operators-sensors-and-scheduling)  
8. [Airflow QA Validation Scenarios](#8-airflow-qa-validation-scenarios)  
9. [Apache Oozie](#9-apache-oozie)  
10. [Kafka and Streaming Pipelines](#10-kafka-and-streaming-pipelines)  
11. [Kafka QA Validation Scenarios](#11-kafka-qa-validation-scenarios)  
12. [dbt for Data Transformation and Testing](#12-dbt-for-data-transformation-and-testing)  
13. [dbt QA Testing Scenarios](#13-dbt-qa-testing-scenarios)  
14. [ETL and ELT Concepts](#14-etl-and-elt-concepts)  
15. [Ingestion Tools](#15-ingestion-tools)  
16. [Transformation Tools](#16-transformation-tools)  
17. [Scheduling and Orchestration Tools](#17-scheduling-and-orchestration-tools)  
18. [Monitoring and Observability Tools](#18-monitoring-and-observability-tools)  
19. [Audit Tables, Reject Tables, and Control Tables](#19-audit-tables-reject-tables-and-control-tables)  
20. [CI/CD for Data Pipelines](#20-cicd-for-data-pipelines)  
21. [Cloud Data Pipeline Tools](#21-cloud-data-pipeline-tools)  
22. [Banking Data Pipeline Testing Scenarios](#22-banking-data-pipeline-testing-scenarios)  
23. [Common Pipeline Failures and Debugging](#23-common-pipeline-failures-and-debugging)  
24. [Important Interview Questions and Model Answers](#24-important-interview-questions-and-model-answers)  
25. [Hands-On Practice Tasks](#25-hands-on-practice-tasks)  
26. [10-Day Preparation Plan](#26-10-day-preparation-plan)  
27. [Final Cheat Sheet](#27-final-cheat-sheet)  
28. [Reference Links](#28-reference-links)

---

# 1. How to Use This Guide

This guide is designed for interview preparation for a **Big Data QA / Quality Engineer** role where data pipelines are tested across tools such as:

- Airflow
- Oozie
- Kafka
- dbt
- Spark jobs
- Hive jobs
- Shell scripts
- Python validation scripts
- Cloud pipeline tools
- CI/CD and monitoring systems

For a Big Data QA role, you do not need to be a senior data platform engineer, but you should be able to explain:

- How pipelines are scheduled
- How jobs depend on each other
- How data moves from source to target
- How to validate job success
- How to debug job failures
- How to verify counts, rejects, and audit results
- How to test incremental and batch pipelines
- How to report pipeline quality issues clearly

The goal of this guide is to help you sound practical and interview-ready.

---

# 2. What “Data Pipeline Tools” Means

A **data pipeline** is a sequence of steps that moves and transforms data from one system to another.

Example:

```text
Source System
     ↓
Landing Zone
     ↓
Raw Table
     ↓
Staging Table
     ↓
Transformation Job
     ↓
Curated Target Table
     ↓
Report / Dashboard / Downstream Application
```

A **data pipeline tool** helps with one or more of the following:

| Function | Example Tools |
|---|---|
| Ingestion | Kafka, Sqoop, NiFi, ADF, AWS Glue |
| Storage | HDFS, S3, ADLS, data lake |
| Processing | Spark, Hive, Databricks, SQL engines |
| Transformation | Spark, dbt, SQL, Python |
| Orchestration | Airflow, Oozie, Control-M, Autosys |
| Scheduling | Airflow, Oozie, cron, enterprise schedulers |
| Testing | dbt tests, pytest, SQL validation, Great Expectations |
| Monitoring | Airflow UI, Spark UI, logs, Grafana, CloudWatch |
| CI/CD | Jenkins, GitHub Actions, GitLab CI, Azure DevOps |
| Metadata | Hive Metastore, data catalog, lineage tools |

---

# 3. Why Pipeline Tools Matter in Big Data QA

Pipeline tools matter because data quality is not just about the final table.

A QA engineer must know:

- Did the pipeline start on time?
- Did the correct upstream file arrive?
- Did each task run successfully?
- Did the transformation logic execute correctly?
- Did failed records go to reject tables?
- Did the target table load the correct partition?
- Did audit counts match?
- Did downstream tasks run only after upstream success?
- Did retry behavior work correctly?
- Were alerts triggered for failures?

In banking environments, data pipelines often support:

- Financial reporting
- Risk calculations
- Customer analytics
- Payments
- Fraud detection
- Regulatory reporting
- Transaction monitoring

So pipeline failures can have serious business impact.

---

# 4. End-to-End Data Pipeline Flow

A typical Big Data pipeline may look like this:

```text
1. Source file arrives in landing path
2. Airflow sensor detects the file
3. Shell/Python task validates file presence and size
4. Spark job loads file into staging
5. Hive/Spark SQL validates schema and raw counts
6. Transformation job applies business rules
7. Good records load to target table
8. Bad records load to reject table
9. Audit table records source, target, and reject counts
10. Downstream job/report starts after successful validation
11. Alerts are sent if any task fails
```

---

## QA checkpoints

| Pipeline Stage | QA Validation |
|---|---|
| Source | File/table exists |
| Landing | File size, file count, naming convention |
| Raw | Schema, delimiter, record count |
| Staging | Nulls, duplicates, invalid records |
| Transform | Business logic, joins, derivations |
| Target | Count, sum, source-to-target mapping |
| Reject | Invalid records captured with reason |
| Audit | Counts and job status are correct |
| Downstream | Data is available for dependent jobs |

---

# 5. Main Categories of Data Pipeline Tools

## 1. Orchestration tools

Used to schedule and coordinate jobs.

Examples:

- Apache Airflow
- Apache Oozie
- Control-M
- Autosys
- Azure Data Factory
- AWS Step Functions
- Databricks Workflows

---

## 2. Ingestion tools

Used to bring data from source systems.

Examples:

- Kafka
- Sqoop
- NiFi
- Flume
- AWS Glue
- Azure Data Factory
- Snowpipe

---

## 3. Processing and transformation tools

Used to transform raw data into usable data.

Examples:

- Spark
- Hive
- dbt
- SQL
- Python
- Databricks
- Snowflake Tasks

---

## 4. Testing and validation tools

Used to validate data quality.

Examples:

- SQL validation scripts
- Python scripts
- PySpark validation
- dbt tests
- Great Expectations
- pytest
- custom QA frameworks

---

## 5. Monitoring and logging tools

Used to observe pipeline health.

Examples:

- Airflow UI
- Spark UI
- YARN logs
- CloudWatch
- Azure Monitor
- Splunk
- Grafana
- Prometheus
- Datadog

---

# 6. Apache Airflow

Apache Airflow is a workflow orchestration tool used to define, schedule, and monitor data pipelines.

Airflow pipelines are written as Python code.

The core Airflow concept is the **DAG**, which stands for Directed Acyclic Graph.

A DAG defines:

- What tasks run
- In what order they run
- When they run
- What happens if they fail
- How many retries are allowed
- Which tasks depend on other tasks

---

## Why Airflow is important for Big Data QA

Airflow is commonly used to orchestrate:

- File arrival checks
- Spark jobs
- Hive jobs
- Python validation scripts
- SQL validation scripts
- dbt runs
- Data quality checks
- Email/Slack alerts
- Downstream dependency triggers

---

## Simple Airflow DAG example

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="daily_transaction_pipeline",
    start_date=datetime(2026, 5, 25),
    schedule="@daily",
    catchup=False
) as dag:

    check_file = BashOperator(
        task_id="check_file_exists",
        bash_command="hdfs dfs -test -e /landing/transactions/txn_{{ ds }}.csv"
    )

    run_spark_job = BashOperator(
        task_id="run_spark_transformation",
        bash_command="spark-submit /jobs/transaction_transform.py {{ ds }}"
    )

    validate_target = BashOperator(
        task_id="validate_target_table",
        bash_command="python /qa/validate_transaction_counts.py {{ ds }}"
    )

    check_file >> run_spark_job >> validate_target
```

---

## Interview explanation

> Airflow is used to orchestrate workflows. A DAG defines task dependencies and scheduling. Each task can run a shell command, Python script, Spark job, Hive query, dbt command, or other operation. In QA, I use Airflow to understand task order, validate job status, check retries, review logs, and verify that data quality checks run before downstream tasks.

---

# 7. Airflow DAGs, Tasks, Operators, Sensors, and Scheduling

## DAG

A DAG is a workflow definition.

```text
check_file → load_staging → transform_data → validate_target → notify
```

A DAG must not have cycles.

For example, this is invalid:

```text
A → B → C → A
```

---

## Task

A task is one unit of work inside a DAG.

Examples:

- Check file exists
- Run Spark job
- Run Hive query
- Run dbt test
- Send notification

---

## Operator

An operator defines what a task does.

Common Airflow operators:

| Operator | Use |
|---|---|
| BashOperator | Run shell command |
| PythonOperator | Run Python function |
| EmptyOperator | Placeholder task |
| EmailOperator | Send email |
| SparkSubmitOperator | Submit Spark job |
| HiveOperator | Run Hive query |
| SqlOperator | Run SQL |
| KubernetesPodOperator | Run task in Kubernetes pod |

---

## Sensor

A sensor waits for something.

Examples:

- Wait for file
- Wait for external DAG
- Wait for HDFS path
- Wait for S3 object
- Wait for SQL condition

Example concept:

```text
FileSensor waits until the expected input file arrives.
```

QA use:

> If a DAG is stuck, I would check whether a sensor is waiting for a file or external dependency.

---

## Scheduling

Airflow DAGs can be scheduled using cron-style expressions or presets.

Examples:

```python
schedule="@daily"
```

```python
schedule="0 2 * * *"
```

This means run every day at 2:00 AM.

---

## Retries

Retries help recover from temporary failures.

Example:

```python
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=10)
}
```

QA point:

> For intermittent upstream or network failures, retries may be expected. But if the same validation fails repeatedly, it likely indicates a real data issue.

---

## catchup

If `catchup=True`, Airflow may run missed historical DAG runs.  
If `catchup=False`, it usually runs only current/future scheduled runs.

QA point:

> I would be careful with catchup because it can trigger multiple historical runs and load old partitions unexpectedly if not configured properly.

---

# 8. Airflow QA Validation Scenarios

## Scenario 1: DAG failed

Checklist:

- Which task failed?
- What was the error message?
- Did upstream tasks succeed?
- Did the task retry?
- Was the input file available?
- Did the Spark/Hive job fail?
- Did the validation script fail?
- Was the failure data-related or environment-related?

---

## Scenario 2: File sensor stuck

Possible causes:

- File did not arrive
- File name has wrong date
- File is in the wrong path
- Permission issue
- Sensor timeout too short
- Upstream source system delayed

Validation command:

```bash
hdfs dfs -ls /landing/transactions/
hdfs dfs -test -e /landing/transactions/txn_20260525.csv
```

---

## Scenario 3: Spark task failed

Check:

- Airflow task logs
- Spark application ID
- YARN logs or Spark UI
- Input path
- Schema issue
- Memory issue
- Bad records
- Permissions
- Output path already exists
- Data skew

---

## Scenario 4: Validation task failed

Example validation failure:

```text
Source count = 1,000,000
Target count = 999,500
Reject count = 300
```

Expected formula:

```text
source_count = target_count + reject_count
```

Here:

```text
1,000,000 != 999,500 + 300
```

Mismatch:

```text
200 records missing or unaccounted for
```

QA response:

> I would check audit tables, reject tables, source-to-target anti-joins, and job logs to identify where the 200 records were lost.

---

## Scenario 5: Downstream task ran even when validation failed

This is a serious orchestration bug.

Expected behavior:

```text
validation_task fails → downstream publishing/reporting task should not run
```

QA should check:

- Airflow dependencies
- Trigger rules
- Task status
- Whether validation script exits with non-zero exit code
- Whether failure was swallowed inside script

Example Python issue:

```python
try:
    validate_counts()
except Exception as e:
    print(e)
```

This is bad if the script does not exit with failure.

Better:

```python
import sys

try:
    validate_counts()
except Exception as e:
    print(e)
    sys.exit(1)
```

---

# 9. Apache Oozie

Apache Oozie is an older Hadoop workflow scheduler.

It is less common in modern new projects than Airflow, but many enterprise Hadoop environments still use it.

---

## Oozie concepts

| Concept | Meaning |
|---|---|
| Workflow | Defines job sequence |
| Coordinator | Schedules workflows based on time/data availability |
| Bundle | Groups multiple coordinators |
| Action | One job step |
| Control node | Start, end, decision, fork, join, kill |
| SLA | Service-level agreement for workflow timing |

---

## Oozie action types

Oozie can run:

- MapReduce jobs
- Pig jobs
- Hive jobs
- Sqoop jobs
- Shell scripts
- Java actions
- Spark jobs, depending on setup

---

## Oozie QA relevance

A QA engineer may need to check:

- Did the workflow start?
- Which action failed?
- Did the coordinator trigger at the correct time?
- Was input data available?
- Did retry or rerun happen?
- Did downstream action run only after upstream success?
- Did SLA miss occur?

---

## Airflow vs Oozie

| Feature | Airflow | Oozie |
|---|---|---|
| Workflow definition | Python DAGs | XML workflows |
| Modern usage | Very common | Legacy Hadoop environments |
| UI | Airflow UI | Oozie UI / CLI |
| Flexibility | High | More Hadoop-specific |
| QA focus | DAG/task status, logs | Workflow/action status, logs |

---

# 10. Kafka and Streaming Pipelines

Kafka is a distributed event streaming platform.

It is used to move events between systems in near real time.

---

## Kafka core concepts

| Concept | Meaning |
|---|---|
| Producer | Application that writes events |
| Consumer | Application that reads events |
| Topic | Named stream of events |
| Partition | Split of a topic for parallel processing |
| Broker | Kafka server |
| Consumer group | Group of consumers sharing work |
| Offset | Position of a consumer in a topic |
| Retention | How long messages are kept |
| Schema Registry | Stores schemas for structured messages |

---

## Simple Kafka flow

```text
Transaction App
     ↓ producer
Kafka Topic: transactions
     ↓ consumer
Spark Streaming / Flink / Consumer App
     ↓
Target Table / Data Lake
```

---

## QA focus in Kafka pipelines

QA may validate:

- Is the producer sending messages?
- Is the topic receiving messages?
- Are messages in the expected schema?
- Are consumers reading messages?
- Is consumer lag increasing?
- Are duplicate messages created?
- Are messages lost?
- Are late events handled?
- Are invalid events sent to a dead-letter topic?
- Are offsets committed correctly?

---

## Consumer lag

Consumer lag means the consumer is behind the producer.

Example:

```text
Latest topic offset: 10,000
Consumer committed offset: 9,500
Consumer lag: 500
```

QA interpretation:

> If lag keeps increasing, the consumer may not be processing events fast enough or may be failing.

---

## Dead-letter topic

A dead-letter topic stores messages that failed processing.

Example:

```text
transactions_raw
transactions_processed
transactions_dead_letter
```

QA should validate:

- Invalid messages are routed to dead-letter topic.
- Valid messages are not sent to dead-letter topic.
- Dead-letter records have failure reasons.
- Counts reconcile between input, output, and failed records.

---

# 11. Kafka QA Validation Scenarios

## Scenario 1: Message count validation

Example:

```text
Produced events: 100,000
Processed events: 99,800
Dead-letter events: 150
```

Expected:

```text
produced = processed + dead_letter + pending
```

If no pending messages are expected:

```text
100,000 should equal 99,800 + 150
```

Mismatch:

```text
50 events missing
```

---

## Scenario 2: Duplicate event check

SQL on target table:

```sql
SELECT event_id, COUNT(*) AS duplicate_count
FROM transaction_event_target
WHERE business_date = '2026-05-25'
GROUP BY event_id
HAVING COUNT(*) > 1;
```

---

## Scenario 3: Schema validation

Check required fields:

```sql
SELECT COUNT(*) AS invalid_record_count
FROM transaction_event_stage
WHERE event_id IS NULL
   OR account_id IS NULL
   OR amount IS NULL
   OR event_timestamp IS NULL;
```

---

## Scenario 4: Late-arriving events

Check events loaded after expected processing window:

```sql
SELECT *
FROM transaction_event_target
WHERE event_timestamp < TIMESTAMP '2026-05-25 00:00:00'
  AND load_timestamp >= TIMESTAMP '2026-05-26 00:00:00';
```

---

## Scenario 5: Consumer lag issue

QA debugging:

- Check if consumer application is running.
- Check if consumer group has lag.
- Check processing errors.
- Check schema compatibility.
- Check target database availability.
- Check if partitions are balanced across consumers.
- Check if downstream writes are slow.

---

# 12. dbt for Data Transformation and Testing

dbt stands for data build tool.

dbt is commonly used for transforming data in a warehouse or lakehouse using SQL.

dbt helps with:

- SQL models
- Data tests
- Documentation
- Lineage
- Incremental models
- Snapshots
- Source freshness checks

---

## dbt project concepts

| Concept | Meaning |
|---|---|
| Model | SQL transformation that creates a table/view |
| Source | Raw input table |
| Seed | Static CSV data loaded into warehouse |
| Snapshot | Tracks historical changes |
| Test | Data quality assertion |
| Macro | Reusable SQL logic |
| Documentation | Generated docs and lineage |
| Run | Build models |
| Test | Validate models |

---

## Example dbt model

File:

```text
models/customer_active.sql
```

SQL:

```sql
SELECT
    customer_id,
    email,
    status,
    updated_timestamp
FROM {{ source('raw', 'customer') }}
WHERE status = 'ACTIVE'
```

---

## Example dbt test YAML

```yaml
version: 2

models:
  - name: customer_active
    columns:
      - name: customer_id
        tests:
          - not_null
          - unique

      - name: email
        tests:
          - not_null
```

---

## Common dbt commands

```bash
dbt debug
dbt compile
dbt run
dbt test
dbt build
dbt docs generate
dbt docs serve
```

---

## QA relevance

dbt is useful for QA because it makes data tests part of the pipeline.

Instead of manually checking every time, tests can run automatically.

Examples:

- Primary key should be unique.
- Mandatory columns should not be null.
- Source values should match accepted values.
- Relationships should be valid.
- Source data should be fresh.

---

# 13. dbt QA Testing Scenarios

## Scenario 1: Unique key test

```yaml
models:
  - name: transaction_fact
    columns:
      - name: transaction_id
        tests:
          - unique
          - not_null
```

Equivalent SQL idea:

```sql
SELECT transaction_id
FROM transaction_fact
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Scenario 2: Accepted values test

```yaml
models:
  - name: transaction_fact
    columns:
      - name: currency_code
        tests:
          - accepted_values:
              values: ['CAD', 'USD', 'EUR', 'GBP']
```

Equivalent SQL idea:

```sql
SELECT *
FROM transaction_fact
WHERE currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP');
```

---

## Scenario 3: Relationship test

```yaml
models:
  - name: transaction_fact
    columns:
      - name: customer_id
        tests:
          - relationships:
              to: ref('customer_dim')
              field: customer_id
```

Equivalent SQL idea:

```sql
SELECT t.customer_id
FROM transaction_fact t
LEFT JOIN customer_dim c
    ON t.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

---

## Scenario 4: Source freshness

Example purpose:

> Check if source data arrived within the expected time window.

QA interpretation:

- If freshness fails, upstream data may be delayed.
- Downstream models should not be trusted until source freshness is resolved.

---

## Scenario 5: dbt test failure

Debug steps:

1. Identify failed model/test.
2. Open compiled SQL.
3. Run the failure query manually.
4. Inspect failing rows.
5. Check upstream source freshness.
6. Check recent code changes.
7. Check incremental model behavior.
8. Check whether test threshold or logic changed.

---

# 14. ETL and ELT Concepts

## ETL

ETL means:

```text
Extract → Transform → Load
```

Data is transformed before loading into the target system.

Example:

```text
Source DB → Spark transformation → Hive target table
```

---

## ELT

ELT means:

```text
Extract → Load → Transform
```

Raw data is loaded first, then transformed inside the warehouse/lakehouse.

Example:

```text
Source DB → Snowflake raw table → dbt transformations → curated table
```

---

## ETL vs ELT

| Area | ETL | ELT |
|---|---|---|
| Transform happens | Before target load | After raw load |
| Common tools | Informatica, Spark, DataStage | dbt, Snowflake, BigQuery |
| Good for | Controlled transformations before load | Cloud warehouses/lakehouses |
| QA focus | Validate staging and target | Validate raw, model, and curated layers |

---

## QA answer

> In ETL, I validate extract, transform, and load stages separately. In ELT, I also validate raw ingestion first, then transformation models in the warehouse or lakehouse. In both cases, I check counts, schema, nulls, duplicates, business rules, rejects, and audit tables.

---

# 15. Ingestion Tools

Ingestion tools bring data into the data platform.

---

## Common ingestion patterns

| Pattern | Example |
|---|---|
| Batch file ingestion | Daily CSV file to HDFS |
| Database ingestion | RDBMS table to data lake |
| Streaming ingestion | Kafka topic to Spark Streaming |
| API ingestion | REST API to raw table |
| CDC ingestion | Changed records from source DB |

---

## Sqoop

Sqoop is used to transfer data between relational databases and Hadoop.

Example use:

```text
Oracle / SQL Server → HDFS / Hive
```

QA checks:

- Source database count
- Imported HDFS file count
- Target Hive table count
- Column mapping
- Null and duplicate checks
- Incremental import logic

---

## NiFi

Apache NiFi is a data flow automation tool.

It can route, transform, and manage data movement.

QA checks:

- FlowFile count
- Processor success/failure
- Back pressure
- Queue buildup
- Failed relationships
- Provenance events

---

## Cloud ingestion examples

| Cloud | Tools |
|---|---|
| AWS | Glue, DMS, Kinesis, S3 events |
| Azure | Azure Data Factory, Event Hubs, Data Lake |
| GCP | Dataflow, Pub/Sub, BigQuery transfers |

---

# 16. Transformation Tools

Transformation tools convert raw data into business-ready data.

---

## Spark

Used for large-scale transformation.

Examples:

- Join transactions with customers
- Calculate derived fields
- Filter invalid records
- Create aggregates
- Write partitioned Parquet tables

QA checks:

- Transformation logic
- Input/output counts
- Duplicate creation
- Join mismatches
- Rejected records
- Partition output
- Spark logs

---

## Hive SQL

Used for SQL-based transformation on Hadoop data.

QA checks:

- Hive table counts
- Partition loads
- Join logic
- Aggregations
- Nulls and invalid values
- File format and location

---

## dbt

Used for SQL transformation in modern warehouses/lakehouses.

QA checks:

- Model output
- dbt tests
- Source freshness
- Incremental model correctness
- Lineage impact
- Documentation

---

## Python

Used for scripting and validation.

QA checks:

- File validation scripts
- API validation
- Data comparison
- Audit report generation
- Pipeline utility scripts

---

# 17. Scheduling and Orchestration Tools

## Airflow

Modern, Python-based orchestrator.

QA focus:

- DAG success/failure
- Task dependencies
- Sensors
- Retries
- Logs
- SLA misses
- Backfills
- Trigger rules

---

## Oozie

Hadoop-native workflow scheduler.

QA focus:

- Workflow actions
- Coordinators
- Input data triggers
- Hadoop job status
- Reruns

---

## Control-M and Autosys

Enterprise schedulers often used in banks.

QA focus:

- Job calendars
- Job dependencies
- Return codes
- Failure alerts
- Reruns
- Batch windows
- Upstream/downstream dependencies

---

## cron

Basic Unix scheduler.

Example:

```bash
0 2 * * * /home/qa/scripts/daily_validation.sh
```

QA focus:

- Did script run?
- Was log generated?
- Did exit code indicate success?
- Were alerts sent?

---

# 18. Monitoring and Observability Tools

Monitoring helps answer:

- Did the job run?
- Did it complete successfully?
- How long did it take?
- Did record counts change unexpectedly?
- Did error rates increase?
- Is consumer lag growing?
- Are tasks retrying?
- Is data delayed?

---

## Common monitoring sources

| Tool / Source | What to Check |
|---|---|
| Airflow UI | DAG/task status, logs, retries |
| Spark UI | stages, tasks, shuffle, failed jobs |
| YARN logs | cluster job logs |
| Hive logs | query failures |
| Kafka monitoring | consumer lag, topic throughput |
| Splunk | application logs |
| Grafana | dashboards and metrics |
| Prometheus | time-series metrics |
| CloudWatch | AWS logs/metrics |
| Azure Monitor | Azure pipeline logs |
| Datadog | service and pipeline observability |

---

## Common metrics

| Metric | Why It Matters |
|---|---|
| Job duration | Detect slow pipeline |
| Failure count | Detect unstable jobs |
| Retry count | Detect intermittent issues |
| Source count | Validate input volume |
| Target count | Validate output volume |
| Reject count | Detect bad data |
| Consumer lag | Detect streaming delay |
| File arrival time | Detect upstream delay |
| Partition count | Validate expected load |
| SLA miss | Detect business impact |

---

# 19. Audit Tables, Reject Tables, and Control Tables

These are some of the most important pipeline QA assets.

---

## Audit table

An audit table stores job execution details.

Example columns:

```text
job_name
batch_id
source_name
target_name
source_count
target_count
reject_count
job_status
start_time
end_time
error_message
created_by
created_timestamp
```

---

## Audit validation query

```sql
SELECT
    job_name,
    batch_id,
    source_count,
    target_count,
    reject_count,
    CASE
        WHEN source_count = target_count + reject_count THEN 'PASS'
        ELSE 'FAIL'
    END AS reconciliation_status
FROM etl_audit
WHERE batch_id = 'BATCH_20260525';
```

---

## Reject table

A reject table stores invalid records.

Example columns:

```text
batch_id
record_id
source_file
reject_reason
raw_record
created_timestamp
```

---

## Reject reason distribution

```sql
SELECT reject_reason, COUNT(*) AS reject_count
FROM reject_records
WHERE batch_id = 'BATCH_20260525'
GROUP BY reject_reason
ORDER BY reject_count DESC;
```

---

## Control table

A control table may store pipeline parameters.

Example columns:

```text
pipeline_name
source_path
target_table
business_date
batch_id
expected_file_count
expected_record_count
is_active
```

QA checks:

- Correct business date
- Correct source path
- Correct target table
- Expected file count
- Pipeline active/inactive flag
- Parameter changes

---

# 20. CI/CD for Data Pipelines

CI/CD means continuous integration and continuous deployment.

For data pipelines, CI/CD may include:

- Code checkout from Git
- Unit tests
- SQL linting
- Python linting
- dbt compile
- dbt test
- Airflow DAG validation
- Deployment to dev/QA/prod
- Smoke tests after deployment

---

## Tools

| Tool | Use |
|---|---|
| Git | Version control |
| Jenkins | Build/deployment automation |
| GitHub Actions | CI/CD workflows |
| GitLab CI | CI/CD workflows |
| Azure DevOps | Pipelines and release management |
| Docker | Containerization |
| Kubernetes/OpenShift | Container orchestration |

---

## QA role in CI/CD

A QA engineer may validate:

- Pipeline code deployed correctly
- Airflow DAG imports without errors
- dbt tests pass
- Spark job parameters are correct
- Environment-specific configs are correct
- Secrets are not hardcoded
- Rollback plan exists
- Smoke tests pass after deployment

---

## Airflow DAG validation example

```bash
python -m py_compile dags/daily_transaction_pipeline.py
```

Or in Airflow environment:

```bash
airflow dags list
airflow tasks list daily_transaction_pipeline
```

---

## dbt validation example

```bash
dbt debug
dbt compile
dbt test
```

---

# 21. Cloud Data Pipeline Tools

Modern data platforms may use cloud-based pipeline tools.

---

## AWS examples

| Tool | Use |
|---|---|
| S3 | Data lake storage |
| Glue | ETL/catalog jobs |
| Step Functions | Workflow orchestration |
| Lambda | Serverless event processing |
| Kinesis | Streaming ingestion |
| DMS | Database migration / CDC |
| CloudWatch | Logs and monitoring |
| Redshift | Data warehouse |

---

## Azure examples

| Tool | Use |
|---|---|
| ADLS | Data lake storage |
| Azure Data Factory | Pipeline orchestration |
| Synapse | Analytics platform |
| Event Hubs | Streaming ingestion |
| Databricks | Spark processing |
| Azure Monitor | Logs and metrics |

---

## GCP examples

| Tool | Use |
|---|---|
| Cloud Storage | Object storage |
| Dataflow | Stream/batch processing |
| Pub/Sub | Messaging/streaming |
| BigQuery | Data warehouse |
| Composer | Managed Airflow |
| Cloud Logging | Logs |

---

## QA approach is similar across clouds

Regardless of platform, validate:

- Source arrival
- Pipeline status
- Record counts
- Schema
- Data quality rules
- Rejects/errors
- Audit logs
- Target availability
- Downstream impact

---

# 22. Banking Data Pipeline Testing Scenarios

## Scenario 1: Daily transaction file pipeline

Flow:

```text
Transaction file → HDFS landing → Spark transform → Hive target → Report
```

QA checks:

```bash
hdfs dfs -test -e /landing/transactions/txn_20260525.csv
hdfs dfs -cat /landing/transactions/txn_20260525.csv | wc -l
```

SQL checks:

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';

SELECT currency_code, SUM(amount)
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY currency_code;
```

---

## Scenario 2: Customer dimension pipeline

Checks:

- One current record per customer
- No null customer ID
- Valid status values
- Source-to-target mapping
- SCD Type 2 history if applicable

```sql
SELECT customer_id, COUNT(*)
FROM customer_dim
WHERE is_current = 'Y'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Scenario 3: Kafka payment events pipeline

Checks:

- Topic receives events
- Consumer is running
- Consumer lag is stable
- Processed count reconciles
- Duplicate event IDs are not created
- Invalid messages go to dead-letter topic

```sql
SELECT event_id, COUNT(*)
FROM payment_event_target
WHERE business_date = '2026-05-25'
GROUP BY event_id
HAVING COUNT(*) > 1;
```

---

## Scenario 4: dbt transformation pipeline

Checks:

- `dbt run` completed
- `dbt test` passed
- Source freshness passed
- Lineage impact reviewed
- Model row counts expected
- Incremental model did not miss records

```bash
dbt run --select transaction_mart
dbt test --select transaction_mart
```

---

## Scenario 5: Regulatory reporting pipeline

Checks:

- Source count reconciles with target
- Amount totals reconcile by currency
- No duplicate business keys
- Audit table shows success
- Rejected records are explainable
- Sensitive information is controlled
- Downstream report totals match curated table

---

# 23. Common Pipeline Failures and Debugging

## Failure 1: Input file missing

Symptoms:

- Sensor stuck
- File check failed
- Source count zero

Debug:

```bash
hdfs dfs -ls /landing/source/
hdfs dfs -test -e /landing/source/file_20260525.csv
```

Possible causes:

- Upstream delay
- Wrong filename
- Wrong date
- File delivered to wrong path
- Permissions issue

---

## Failure 2: Schema mismatch

Symptoms:

- Spark read failure
- Hive columns shifted
- Nulls suddenly increased
- dbt model failure

Debug:

```python
df.printSchema()
df.show(5, truncate=False)
```

SQL:

```sql
DESCRIBE FORMATTED table_name;
```

Possible causes:

- New column added
- Column order changed
- Delimiter changed
- Header changed
- Data type changed

---

## Failure 3: Count mismatch

Debug order:

1. Source count
2. Landing count
3. Staging count
4. Target count
5. Reject count
6. Audit table
7. Duplicate checks
8. Missing record anti-join
9. Transformation filters
10. Logs

---

## Failure 4: Duplicate records

Possible causes:

- Pipeline rerun without cleanup
- Incremental logic issue
- Merge/upsert failure
- Duplicate source data
- Kafka replay
- Bad primary key logic

SQL:

```sql
SELECT business_key, COUNT(*)
FROM target_table
GROUP BY business_key
HAVING COUNT(*) > 1;
```

---

## Failure 5: Airflow DAG stuck

Possible causes:

- Sensor waiting
- Upstream dependency not complete
- Pool/concurrency limit
- Scheduler issue
- External task not completed
- Task queued but no worker available

QA checks:

- DAG status
- Task status
- Logs
- Sensor timeout
- Upstream DAG
- Worker availability
- Schedule time

---

## Failure 6: dbt test failed

Possible causes:

- Real data quality issue
- Source data changed
- Test threshold too strict
- Model logic changed
- Incremental model issue
- Upstream freshness failure

Debug:

```bash
dbt test --select model_name
dbt compile
```

Then inspect compiled SQL and failing rows.

---

## Failure 7: Kafka consumer lag

Possible causes:

- Consumer app down
- Processing too slow
- Target system slow
- Bad messages causing retries
- More producers than expected
- Partition imbalance

QA checks:

- Consumer group status
- Error logs
- Dead-letter topic
- Throughput
- Target write performance

---

# 24. Important Interview Questions and Model Answers

## Q1. What is a data pipeline?

**Answer:**  
A data pipeline is a series of steps that moves data from source to target. It may include ingestion, validation, transformation, loading, audit logging, and downstream publishing.

---

## Q2. What is orchestration?

**Answer:**  
Orchestration is the coordination of pipeline tasks in the correct order with dependencies, scheduling, retries, and monitoring. Airflow and Oozie are examples of orchestration tools.

---

## Q3. What is Airflow?

**Answer:**  
Airflow is a workflow orchestration tool. It uses Python DAGs to define tasks, dependencies, schedules, retries, and monitoring. In Big Data QA, I use Airflow to check pipeline status, task logs, dependencies, failed tasks, and validation task results.

---

## Q4. What is a DAG?

**Answer:**  
A DAG is a Directed Acyclic Graph. It defines the order of tasks in a workflow without circular dependencies.

---

## Q5. What is an Airflow sensor?

**Answer:**  
A sensor is a task that waits for a condition, such as a file arriving or another job completing. If a DAG is stuck, I would check whether a sensor is waiting for an input file or external dependency.

---

## Q6. How do you debug a failed Airflow task?

**Answer:**  
I would check the failed task logs, upstream dependencies, input files, parameters, retry history, error message, and whether the failure happened in a shell command, Python script, Spark job, Hive query, or validation check.

---

## Q7. What is Kafka?

**Answer:**  
Kafka is a distributed event streaming platform. Producers write events to topics, and consumers read events from topics. It is commonly used for real-time or near-real-time data pipelines.

---

## Q8. What is consumer lag?

**Answer:**  
Consumer lag is the difference between the latest offset in a Kafka topic and the offset processed by a consumer group. Increasing lag means the consumer is falling behind.

---

## Q9. What is dbt?

**Answer:**  
dbt is a SQL-based data transformation tool used to build models, run tests, document datasets, and manage data lineage. It is commonly used in modern analytics pipelines.

---

## Q10. What are dbt tests?

**Answer:**  
dbt tests are assertions about data quality. For example, a column should be unique, not null, have accepted values, or maintain a relationship with another table.

---

## Q11. What is the difference between ETL and ELT?

**Answer:**  
ETL transforms data before loading it into the target. ELT loads raw data first and transforms it inside the warehouse or lakehouse. QA checks are similar, but ELT usually requires stronger raw-layer and model-layer validation.

---

## Q12. How do you validate a pipeline?

**Answer:**  
I validate a pipeline in layers. First, I check file arrival or source availability. Then I validate raw counts, schema, duplicates, nulls, and invalid values. After transformation, I compare source and target counts, sums, and key fields. I also check reject tables, audit tables, logs, partitions, and downstream dependencies.

---

## Q13. How do you handle a count mismatch?

**Answer:**  
I compare source, staging, target, and reject counts. Then I group by date, source system, transaction type, or currency to isolate the mismatch. After that, I use anti-joins to find missing records and review logs for filters, rejects, or failed writes.

---

## Q14. What is a dead-letter queue or dead-letter topic?

**Answer:**  
It stores records or messages that failed processing. QA should validate that invalid records go to the dead-letter location with a clear failure reason and that valid records are not incorrectly rejected.

---

## Q15. What should happen if a validation task fails?

**Answer:**  
The pipeline should stop downstream publishing or reporting tasks, mark the job as failed, log the failure reason, and trigger an alert. The validation script should return a non-zero exit code so the scheduler recognizes the failure.

---

# 25. Hands-On Practice Tasks

## Task 1: Create a simple Airflow DAG

Practice creating a DAG with three tasks:

```text
check_file → run_job → validate_output
```

---

## Task 2: Write a file check command

```bash
hdfs dfs -test -e /landing/customer/customer_20260525.csv
echo $?
```

---

## Task 3: Write an audit reconciliation query

```sql
SELECT
    batch_id,
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

## Task 4: Write a dbt uniqueness test

```yaml
models:
  - name: transaction_fact
    columns:
      - name: transaction_id
        tests:
          - unique
          - not_null
```

---

## Task 5: Write a Kafka duplicate validation SQL query

```sql
SELECT event_id, COUNT(*)
FROM event_target
WHERE business_date = '2026-05-25'
GROUP BY event_id
HAVING COUNT(*) > 1;
```

---

## Task 6: Debug a failed DAG scenario

Given:

```text
check_file: success
load_staging: success
transform_data: failed
validate_target: skipped
publish_report: skipped
```

Expected explanation:

> The pipeline stopped at transformation. I would check transform task logs, Spark/Hive error, input schema, bad records, output path, permissions, and recent code/config changes. Skipping validation and publishing is expected because the transform did not complete.

---

## Task 7: Debug incorrect downstream execution

Given:

```text
validate_target: failed
publish_report: success
```

Expected explanation:

> This is a pipeline control issue. The downstream report should not publish if validation fails. I would check Airflow trigger rules, task dependencies, and whether the validation script returned a non-zero exit code.

---

# 26. 10-Day Preparation Plan

## Day 1: Pipeline fundamentals

Study:

- What is a data pipeline?
- ETL vs ELT
- Batch vs streaming
- Ingestion, transformation, loading
- Audit and reject concepts

Goal: Explain pipeline flow clearly.

---

## Day 2: Airflow basics

Study:

- DAG
- Task
- Operator
- Sensor
- Schedule
- Retry
- Logs
- Task dependencies

Goal: Explain how Airflow orchestrates pipelines.

---

## Day 3: Airflow QA scenarios

Practice:

- Failed DAG debugging
- File sensor stuck
- Validation task failed
- Retry behavior
- Downstream dependency issue
- Backfill/catchup issue

Goal: Answer operational interview questions.

---

## Day 4: Oozie and enterprise schedulers

Study:

- Oozie workflows
- Coordinators
- Control-M
- Autosys
- Return codes
- Batch windows

Goal: Understand legacy and enterprise scheduling concepts.

---

## Day 5: Kafka basics

Study:

- Producer
- Consumer
- Topic
- Partition
- Offset
- Consumer group
- Consumer lag
- Dead-letter topic

Goal: Explain streaming pipeline testing.

---

## Day 6: Kafka QA scenarios

Practice:

- Duplicate event checks
- Missing event checks
- Lag debugging
- Schema validation
- Dead-letter validation
- Event count reconciliation

Goal: Handle streaming QA questions.

---

## Day 7: dbt basics

Study:

- Models
- Sources
- Tests
- Seeds
- Snapshots
- Documentation
- Lineage
- Incremental models

Goal: Explain dbt in data transformation and QA.

---

## Day 8: Audit, reject, and control tables

Practice SQL for:

- Audit reconciliation
- Reject reason distribution
- Control table parameter validation
- Batch status validation

Goal: Sound strong in enterprise data QA.

---

## Day 9: CI/CD and monitoring

Study:

- Git
- Jenkins
- Azure DevOps
- GitHub Actions
- Airflow deployment
- dbt test in pipeline
- Logs and metrics

Goal: Understand how pipeline code moves to production.

---

## Day 10: Mock interview

Practice answering:

- How do you validate a data pipeline?
- How do you debug a failed DAG?
- What is consumer lag?
- What are dbt tests?
- What is ETL vs ELT?
- What should happen if validation fails?
- How do audit and reject tables help?
- How do you handle count mismatch?

Goal: Give structured, practical answers.

---

# 27. Final Cheat Sheet

## Data pipeline layers

```text
Source
Landing
Raw
Staging
Transform
Target
Reject
Audit
Downstream
```

---

## Most important tools

```text
Airflow
Oozie
Kafka
dbt
Spark
Hive
SQL
Python
Shell scripting
Jenkins
Git
Monitoring/logging tools
```

---

## Airflow terms

```text
DAG
Task
Operator
Sensor
Schedule
Retry
Trigger rule
Backfill
catchup
Task logs
SLA miss
```

---

## Kafka terms

```text
Producer
Consumer
Topic
Partition
Broker
Offset
Consumer group
Consumer lag
Dead-letter topic
Schema Registry
```

---

## dbt terms

```text
Model
Source
Test
Seed
Snapshot
Macro
Lineage
Documentation
Incremental model
Source freshness
```

---

## Pipeline QA checks

```text
File exists
File size
Record count
Schema validation
Null check
Duplicate check
Business rule check
Source-to-target comparison
Reject validation
Audit validation
Partition validation
Downstream dependency check
```

---

## Common SQL checks

```sql
-- Audit reconciliation
SELECT
    source_count,
    target_count,
    reject_count,
    CASE
        WHEN source_count = target_count + reject_count THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM etl_audit
WHERE batch_id = 'BATCH_20260525';

-- Duplicate check
SELECT business_key, COUNT(*)
FROM target_table
GROUP BY business_key
HAVING COUNT(*) > 1;

-- Missing in target
SELECT s.*
FROM source_table s
LEFT JOIN target_table t
    ON s.business_key = t.business_key
WHERE t.business_key IS NULL;
```

---

## Strong interview answer

If asked, **“How do you validate a data pipeline?”**, say:

> I validate a data pipeline in layers. First, I confirm the source data or input file arrived correctly. Then I validate raw and staging counts, schema, duplicates, nulls, and invalid records. After transformation, I compare source and target counts, sums, and key business fields. I also validate reject records, audit tables, target partitions, job logs, and downstream dependencies. If a validation fails, the pipeline should stop downstream tasks, log the reason, and trigger an alert.

---

# 28. Reference Links

Use these for deeper study:

- Apache Airflow Documentation: https://airflow.apache.org/docs/
- Airflow DAGs: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html
- Airflow Tasks: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html
- Apache Kafka Documentation: https://kafka.apache.org/documentation/
- dbt Documentation: https://docs.getdbt.com/
- dbt Data Tests: https://docs.getdbt.com/docs/build/data-tests
- Apache Oozie Documentation: https://oozie.apache.org/docs/
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Apache Hive Documentation: https://hive.apache.org/docs/latest/
