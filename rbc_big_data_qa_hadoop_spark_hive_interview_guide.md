# Hadoop, Spark, Hive, and Big Data Interview Guide  
## For Big Data QA / Quality Engineer Role at RBC

**Target roles:** Big Data QA, Data QA Analyst, ETL QA Engineer, Quality Engineer, Data Validation Engineer  
**Target environment:** Banking / financial services / RBC-style enterprise data platforms  
**Main focus:** Hadoop, HDFS, Hive, Spark, PySpark, Spark SQL, batch pipelines, data validation, reconciliation, performance, and troubleshooting.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)  
2. [What Big Data QA Interviews Usually Test](#2-what-big-data-qa-interviews-usually-test)  
3. [Big Data Basics](#3-big-data-basics)  
4. [Hadoop Ecosystem Overview](#4-hadoop-ecosystem-overview)  
5. [HDFS Concepts](#5-hdfs-concepts)  
6. [Important HDFS Commands](#6-important-hdfs-commands)  
7. [YARN and Cluster Resource Concepts](#7-yarn-and-cluster-resource-concepts)  
8. [MapReduce Basics](#8-mapreduce-basics)  
9. [Hive Overview](#9-hive-overview)  
10. [Hive DDL and Table Concepts](#10-hive-ddl-and-table-concepts)  
11. [Hive Partitions and Bucketing](#11-hive-partitions-and-bucketing)  
12. [Hive File Formats](#12-hive-file-formats)  
13. [Hive SQL for QA Validation](#13-hive-sql-for-qa-validation)  
14. [Spark Overview](#14-spark-overview)  
15. [Spark Architecture](#15-spark-architecture)  
16. [RDD, DataFrame, and Dataset Basics](#16-rdd-dataframe-and-dataset-basics)  
17. [Spark Transformations and Actions](#17-spark-transformations-and-actions)  
18. [Spark SQL](#18-spark-sql)  
19. [PySpark for Big Data QA](#19-pyspark-for-big-data-qa)  
20. [Source-to-Target Testing in Big Data](#20-source-to-target-testing-in-big-data)  
21. [Data Quality Checks](#21-data-quality-checks)  
22. [Batch Pipeline Testing](#22-batch-pipeline-testing)  
23. [Incremental Load and CDC Testing](#23-incremental-load-and-cdc-testing)  
24. [File-Level Validation](#24-file-level-validation)  
25. [Partition-Level Validation](#25-partition-level-validation)  
26. [Reject and Audit Table Validation](#26-reject-and-audit-table-validation)  
27. [Spark Performance Concepts](#27-spark-performance-concepts)  
28. [Hive Performance Concepts](#28-hive-performance-concepts)  
29. [Common Production Issues and Debugging](#29-common-production-issues-and-debugging)  
30. [Banking Big Data QA Scenarios](#30-banking-big-data-qa-scenarios)  
31. [Interview Questions and Model Answers](#31-interview-questions-and-model-answers)  
32. [Hands-On Practice Tasks](#32-hands-on-practice-tasks)  
33. [10-Day Preparation Plan](#33-10-day-preparation-plan)  
34. [Final Cheat Sheet](#34-final-cheat-sheet)  
35. [Reference Links](#35-reference-links)

---

# 1. How to Use This Guide

Use this guide as a practical interview preparation document.

For a Big Data QA role, you do **not** need to sound like a Hadoop administrator or Spark platform engineer. You need to show that you can:

- Understand how large-scale data pipelines work.
- Validate files and tables across Hadoop/Spark/Hive environments.
- Write SQL and PySpark checks.
- Compare source and target data.
- Debug batch failures.
- Understand partitions, file formats, and job logs.
- Explain performance issues at a QA level.
- Think carefully about data accuracy, auditability, and security.

The most important mindset:

> In Big Data QA, the goal is not only to check if a job completed. The goal is to prove that the right data arrived, transformed correctly, loaded completely, reconciled with source, and can be trusted by downstream users.

---

# 2. What Big Data QA Interviews Usually Test

A Big Data QA interview commonly tests five areas:

## 1. Big Data concepts

You should understand:

- What Big Data means.
- Why distributed storage and distributed processing are used.
- What Hadoop, Hive, and Spark do.
- Difference between storage, compute, and query layers.

## 2. HDFS and file validation

You may be asked:

- How do you check if a file exists in HDFS?
- How do you count records in an HDFS file?
- How do you validate a file was generated successfully?
- How do you check HDFS directory size?

## 3. Hive SQL and data validation

You may be asked:

- How do you validate source and target count?
- How do you check duplicates?
- How do you validate partitions?
- What is the difference between managed and external Hive tables?
- What are partitions and buckets?

## 4. Spark and PySpark

You may be asked:

- What is Spark?
- What are transformations and actions?
- What is lazy evaluation?
- How do you read a CSV or Parquet file in PySpark?
- How do you perform count, duplicate, null, and reconciliation checks?

## 5. Data QA thinking

You may be asked:

- How would you test a batch pipeline?
- How would you debug a failed load?
- How would you validate a financial transaction pipeline?
- How do you handle billions of records?
- How do you validate data without comparing every row manually?

---

# 3. Big Data Basics

## What is Big Data?

Big Data refers to datasets that are too large, fast, or complex for traditional systems to process efficiently.

It is often explained using the 5 Vs:

| Term | Meaning | QA Relevance |
|---|---|---|
| Volume | Large amount of data | Need scalable validation |
| Velocity | Data arrives quickly | Need batch or streaming checks |
| Variety | Different formats | Need schema and format validation |
| Veracity | Data quality and trust | Core QA responsibility |
| Value | Useful business insight | QA protects business decisions |

---

## Why do companies use Big Data platforms?

Traditional databases may struggle with:

- Huge files
- Billions of rows
- Semi-structured data
- Distributed processing
- Long-running ETL jobs
- High storage cost
- Large-scale reporting and analytics

Big Data platforms help by distributing storage and computation across multiple machines.

---

## Big Data QA role in simple words

A Big Data QA engineer checks whether large-scale data pipelines produce correct, complete, accurate, and reliable data.

Example:

A source system sends 10 million transaction records. A Spark job transforms the data and loads it into a Hive table.

As QA, you validate:

- Did all 10 million records arrive?
- Did the target table also have 10 million records?
- Were invalid records rejected correctly?
- Did transaction amounts reconcile?
- Were duplicate transaction IDs created?
- Were partitions loaded correctly?
- Did customer/account mapping work correctly?
- Did the job logs show errors?
- Was sensitive data exposed in logs?

---

# 4. Hadoop Ecosystem Overview

Hadoop is not just one tool. It is an ecosystem of tools for distributed storage, processing, querying, and workflow management.

## Common Hadoop ecosystem components

| Component | Purpose |
|---|---|
| HDFS | Distributed file storage |
| YARN | Cluster resource management |
| MapReduce | Distributed batch processing model |
| Hive | SQL layer on top of Hadoop data |
| Spark | Fast distributed processing engine |
| HBase | NoSQL database on Hadoop |
| Sqoop | Data transfer between relational DBs and Hadoop |
| Flume | Log/data ingestion tool |
| Oozie | Workflow scheduler for Hadoop jobs |
| Airflow | Modern workflow orchestration tool |
| Kafka | Streaming/event ingestion platform |
| Zookeeper | Coordination service |
| Ranger/Sentry | Security and authorization |
| Kerberos | Authentication |

---

## Storage vs processing vs query layer

| Layer | Examples | What it does |
|---|---|---|
| Storage | HDFS, S3, ADLS | Stores raw and processed data |
| Processing | Spark, MapReduce | Transforms data |
| Query | Hive, Spark SQL, Trino | Lets users query data |
| Orchestration | Airflow, Oozie | Schedules workflows |
| Metadata | Hive Metastore | Stores table definitions |
| Security | Kerberos, Ranger | Controls access |

---

## Strong interview explanation

> Hadoop provides distributed storage through HDFS and distributed processing through tools like MapReduce and Spark. Hive allows SQL-style querying on data stored in Hadoop. Spark is commonly used for faster transformations and analytics. In Big Data QA, I use these tools to validate file arrival, schema, counts, partitions, duplicates, nulls, business rules, and source-to-target reconciliation.

---

# 5. HDFS Concepts

HDFS stands for **Hadoop Distributed File System**.

It is designed to store very large files across multiple machines.

## Key HDFS terms

| Term | Meaning |
|---|---|
| NameNode | Master service that manages file system metadata |
| DataNode | Worker node that stores actual data blocks |
| Block | A piece of a file stored in HDFS |
| Replication | Copies of blocks stored on different nodes |
| Rack awareness | Data placement strategy across racks |
| High throughput | Optimized for large reads/writes |
| Fault tolerance | System continues if a node fails |

---

## How HDFS stores a file

If a 1 GB file is uploaded to HDFS:

1. HDFS breaks the file into blocks.
2. Blocks are distributed across DataNodes.
3. Each block is replicated.
4. NameNode tracks where each block is stored.
5. Users access the file as if it is one file.

---

## Why HDFS is useful

HDFS is useful because:

- It can store very large files.
- It runs on clusters of commodity hardware.
- It is fault-tolerant through replication.
- It is optimized for high-throughput batch processing.

---

## HDFS is not ideal for everything

HDFS is not ideal for:

- Very small files in huge numbers.
- Low-latency random updates.
- Frequent small writes.
- Traditional OLTP-style workloads.

QA interview point:

> If a pipeline creates many small files, that can hurt Hadoop/Spark performance because metadata and task overhead increase.

---

# 6. Important HDFS Commands

You should be very comfortable with HDFS commands.

Most commands use this pattern:

```bash
hdfs dfs <command> <path>
```

---

## List HDFS files

```bash
hdfs dfs -ls /data/input
```

Long listing:

```bash
hdfs dfs -ls -h /data/input
```

Recursive listing:

```bash
hdfs dfs -ls -R /data/input
```

---

## Check if file exists

```bash
hdfs dfs -test -e /data/input/customer.csv

if [ $? -eq 0 ]; then
    echo "File exists"
else
    echo "File missing"
fi
```

---

## View file content

```bash
hdfs dfs -cat /data/input/customer.csv
```

First part of file:

```bash
hdfs dfs -head /data/input/customer.csv
```

Last part of file:

```bash
hdfs dfs -tail /data/input/customer.csv
```

---

## Count records in HDFS file

```bash
hdfs dfs -cat /data/input/customer.csv | wc -l
```

Count excluding header:

```bash
hdfs dfs -cat /data/input/customer.csv | tail -n +2 | wc -l
```

---

## Check HDFS directory size

```bash
hdfs dfs -du -h /data/input
```

Summary size:

```bash
hdfs dfs -du -s -h /data/input
```

---

## Count files and directories

```bash
hdfs dfs -count /data/input
```

This typically returns directory count, file count, and content size.

---

## Upload local file to HDFS

```bash
hdfs dfs -put customer.csv /data/input/
```

Overwrite if needed:

```bash
hdfs dfs -put -f customer.csv /data/input/
```

---

## Download file from HDFS

```bash
hdfs dfs -get /data/input/customer.csv .
```

---

## Create HDFS directory

```bash
hdfs dfs -mkdir /data/input/new_folder
```

Create parent directories too:

```bash
hdfs dfs -mkdir -p /data/input/new_folder/sub_folder
```

---

## Remove file

```bash
hdfs dfs -rm /data/input/customer.csv
```

Remove folder recursively:

```bash
hdfs dfs -rm -r /data/input/archive
```

Be careful with remove commands in real environments.

---

## Copy inside HDFS

```bash
hdfs dfs -cp /data/input/file1.csv /data/archive/file1.csv
```

---

## Move inside HDFS

```bash
hdfs dfs -mv /data/input/file1.csv /data/archive/file1.csv
```

---

## QA file arrival script

```bash
#!/bin/bash

hdfs_path=$1

if [ -z "$hdfs_path" ]; then
    echo "Usage: $0 <hdfs_path>"
    exit 1
fi

hdfs dfs -test -e "$hdfs_path"

if [ $? -eq 0 ]; then
    echo "PASS: File exists in HDFS: $hdfs_path"
else
    echo "FAIL: File missing in HDFS: $hdfs_path"
    exit 1
fi
```

---

# 7. YARN and Cluster Resource Concepts

YARN stands for **Yet Another Resource Negotiator**.

It manages resources in a Hadoop cluster.

## Important YARN concepts

| Term | Meaning |
|---|---|
| ResourceManager | Cluster-level resource manager |
| NodeManager | Runs on worker nodes and manages resources |
| ApplicationMaster | Manages one application/job |
| Container | Allocated CPU/memory for a task |

---

## Why QA should understand YARN

You may not manage YARN directly, but you should know how it affects jobs:

- Spark jobs request containers.
- Jobs may fail if memory is insufficient.
- Jobs may be stuck if cluster resources are unavailable.
- Logs may be available through YARN.
- Runtime and failure messages can help debugging.

---

## Useful YARN commands

List applications:

```bash
yarn application -list
```

Check application status:

```bash
yarn application -status <application_id>
```

Get logs:

```bash
yarn logs -applicationId <application_id>
```

Kill application:

```bash
yarn application -kill <application_id>
```

QA note: In real projects, do not kill jobs unless you are authorized.

---

# 8. MapReduce Basics

MapReduce is an older distributed processing model in Hadoop.

Spark is more common in modern pipelines, but MapReduce is still useful conceptually.

## MapReduce flow

1. Input data is split.
2. Mapper processes each split.
3. Shuffle sorts/groups intermediate data.
4. Reducer aggregates or combines results.
5. Output is written to HDFS.

---

## Simple example

Input:

```text
apple
orange
apple
banana
orange
apple
```

Mapper emits:

```text
apple, 1
orange, 1
apple, 1
banana, 1
orange, 1
apple, 1
```

Reducer outputs:

```text
apple, 3
orange, 2
banana, 1
```

---

## QA relevance

MapReduce concepts help explain:

- Distributed processing
- Shuffling
- Grouping
- Why large joins and aggregations can be expensive
- Why output is often split into multiple part files

---

# 9. Hive Overview

Hive is a data warehouse system that allows SQL-style queries on data stored in distributed storage.

Hive is useful because many QA engineers already know SQL and can use HiveQL to validate Big Data tables.

---

## Hive components

| Component | Meaning |
|---|---|
| HiveQL | SQL-like language |
| Hive Metastore | Stores table metadata |
| Hive table | Table definition over files |
| Managed table | Hive manages table data |
| External table | Data exists outside Hive ownership |
| Partition | Directory-based data organization |
| SerDe | Serializer/deserializer for reading file format |

---

## Hive vs traditional database

| Traditional database | Hive |
|---|---|
| Good for OLTP and quick row lookups | Good for large-scale batch analytics |
| Data stored in database-managed format | Data stored in HDFS/cloud storage |
| Updates are common | Append/overwrite patterns are common |
| Low latency possible | Higher latency, large scans |
| Indexes commonly used | Partitions and file formats are important |

---

## Strong interview explanation

> Hive allows SQL-style querying on large datasets stored in HDFS or distributed storage. In QA, I use Hive to validate counts, duplicates, nulls, partitions, source-to-target mappings, reject records, and aggregate reconciliations.

---

# 10. Hive DDL and Table Concepts

## Create database

```sql
CREATE DATABASE IF NOT EXISTS qa_demo;
```

Use database:

```sql
USE qa_demo;
```

---

## Create managed table

```sql
CREATE TABLE customer_managed (
    customer_id STRING,
    first_name STRING,
    last_name STRING,
    email STRING,
    status STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
```

For a managed table, Hive owns the data location.

---

## Create external table

```sql
CREATE EXTERNAL TABLE customer_external (
    customer_id STRING,
    first_name STRING,
    last_name STRING,
    email STRING,
    status STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/data/customer/';
```

For an external table, Hive points to an existing data location.

---

## Managed vs external table

| Feature | Managed Table | External Table |
|---|---|---|
| Data ownership | Hive owns data | Data owned outside Hive |
| Drop table behavior | May remove data | Usually removes metadata only |
| Common use | Temporary/internal tables | Raw/staging/enterprise data |
| QA relevance | Be careful before dropping | Common for validating HDFS data |

---

## Show tables

```sql
SHOW TABLES;
```

---

## Describe table

```sql
DESCRIBE customer_external;
```

Detailed description:

```sql
DESCRIBE FORMATTED customer_external;
```

---

## Load data

```sql
LOAD DATA INPATH '/data/input/customer.csv'
INTO TABLE customer_managed;
```

Overwrite:

```sql
LOAD DATA INPATH '/data/input/customer.csv'
OVERWRITE INTO TABLE customer_managed;
```

---

# 11. Hive Partitions and Bucketing

Partitions and buckets are important in Big Data interviews.

---

## What is partitioning?

Partitioning stores data in separate folders based on column values.

Example:

```text
/data/transactions/business_date=2026-05-25/
/data/transactions/business_date=2026-05-26/
```

This allows queries to scan only needed partitions.

---

## Create partitioned table

```sql
CREATE TABLE transaction_fact (
    transaction_id STRING,
    account_id STRING,
    customer_id STRING,
    amount DECIMAL(18,2),
    currency_code STRING
)
PARTITIONED BY (business_date STRING)
STORED AS PARQUET;
```

---

## Insert into partition

```sql
INSERT INTO TABLE transaction_fact
PARTITION (business_date='2026-05-25')
SELECT
    transaction_id,
    account_id,
    customer_id,
    amount,
    currency_code
FROM transaction_stage
WHERE business_date = '2026-05-25';
```

---

## Show partitions

```sql
SHOW PARTITIONS transaction_fact;
```

---

## Query specific partition

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

This is much better than scanning the full table.

---

## Partition validation query

```sql
SELECT
    business_date,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY business_date;
```

---

## What is bucketing?

Bucketing divides data into a fixed number of files based on a hash of a column.

Example:

```sql
CREATE TABLE customer_bucketed (
    customer_id STRING,
    name STRING,
    email STRING
)
CLUSTERED BY (customer_id) INTO 16 BUCKETS
STORED AS ORC;
```

---

## Partitioning vs bucketing

| Concept | Partitioning | Bucketing |
|---|---|---|
| Based on | Column value | Hash of column |
| Physical result | Folders | Fixed number of files |
| Best for | Date, region, batch | Joins, sampling, even distribution |
| Example | `business_date=2026-05-25` | 16 buckets by customer_id |

---

# 12. Hive File Formats

Big Data tables can use different file formats.

## Common formats

| Format | Description | QA Notes |
|---|---|---|
| Text/CSV | Simple row-based format | Easy to inspect, less efficient |
| JSON | Semi-structured data | Schema handling needed |
| Avro | Row-based, schema support | Good for data interchange |
| ORC | Columnar format | Efficient for Hive analytics |
| Parquet | Columnar format | Common with Spark and Hive |
| SequenceFile | Hadoop binary format | Older Hadoop use cases |

---

## Row-based vs columnar

| Type | Examples | Best for |
|---|---|---|
| Row-based | CSV, JSON, Avro | Full-row reads, ingestion |
| Columnar | ORC, Parquet | Analytics, selected columns, compression |

---

## Why file format matters for QA

File format affects:

- Query performance
- Storage size
- Schema evolution
- Compression
- Compatibility between tools
- Whether files can be inspected easily

QA interview point:

> For large analytics tables, Parquet or ORC is usually more efficient than CSV because queries can read only required columns and benefit from compression.

---

# 13. Hive SQL for QA Validation

Hive SQL is very important for Big Data QA.

---

## Count validation

```sql
SELECT COUNT(*) AS record_count
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Duplicate validation

```sql
SELECT transaction_id, COUNT(*) AS duplicate_count
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Null validation

```sql
SELECT COUNT(*) AS null_customer_count
FROM transaction_fact
WHERE business_date = '2026-05-25'
  AND customer_id IS NULL;
```

---

## Invalid currency validation

```sql
SELECT currency_code, COUNT(*) AS invalid_count
FROM transaction_fact
WHERE business_date = '2026-05-25'
  AND currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP')
GROUP BY currency_code;
```

---

## Amount reconciliation

```sql
SELECT
    business_date,
    currency_code,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY business_date, currency_code;
```

---

## Source-to-target count comparison

```sql
SELECT 'source' AS table_name, COUNT(*) AS record_count
FROM source_transaction
WHERE business_date = '2026-05-25'

UNION ALL

SELECT 'target' AS table_name, COUNT(*) AS record_count
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Missing records in target

```sql
SELECT s.transaction_id
FROM source_transaction s
LEFT JOIN transaction_fact t
    ON s.transaction_id = t.transaction_id
   AND t.business_date = '2026-05-25'
WHERE s.business_date = '2026-05-25'
  AND t.transaction_id IS NULL;
```

---

## Field-level mismatch check

```sql
SELECT
    s.transaction_id,
    s.amount AS source_amount,
    t.amount AS target_amount,
    s.currency_code AS source_currency,
    t.currency_code AS target_currency
FROM source_transaction s
JOIN transaction_fact t
    ON s.transaction_id = t.transaction_id
WHERE s.business_date = '2026-05-25'
  AND t.business_date = '2026-05-25'
  AND (
        s.amount <> t.amount
        OR s.currency_code <> t.currency_code
      );
```

---

# 14. Spark Overview

Apache Spark is a distributed analytics engine used for large-scale data processing.

Spark is widely used because it can process data faster than traditional MapReduce for many workloads and supports APIs such as Python, Scala, Java, R, and SQL.

---

## Spark supports

| Component | Purpose |
|---|---|
| Spark Core | Basic distributed processing |
| Spark SQL | SQL and DataFrame processing |
| Structured Streaming | Streaming data processing |
| MLlib | Machine learning library |
| GraphX | Graph processing |
| PySpark | Python API for Spark |

---

## Why Spark is important for QA

Spark is commonly used to:

- Read large files
- Transform data
- Join datasets
- Load Hive tables
- Write Parquet/ORC files
- Run ETL pipelines
- Perform data validations at scale

QA engineers may use Spark to validate data when SQL alone is not enough or when files are too large for local Unix/Python processing.

---

# 15. Spark Architecture

## Main Spark components

| Component | Meaning |
|---|---|
| Driver | Main process that coordinates the Spark application |
| Executor | Worker process that runs tasks |
| Cluster Manager | Allocates resources |
| Task | Smallest unit of work |
| Job | Triggered by an action |
| Stage | Set of tasks separated by shuffle boundaries |
| DAG | Directed Acyclic Graph of execution steps |

---

## Spark execution flow

1. User submits Spark application.
2. Driver starts.
3. Driver requests resources from cluster manager.
4. Executors are launched.
5. Driver builds execution plan.
6. Tasks run on executors.
7. Results are returned or written to storage.

---

## Spark on YARN

In Hadoop environments, Spark can run on YARN.

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4G \
  --num-executors 10 \
  qa_validation_job.py
```

---

## Client mode vs cluster mode

| Mode | Driver runs on | Common use |
|---|---|---|
| Client mode | Machine submitting job | Development/debugging |
| Cluster mode | Cluster node | Production jobs |

---

# 16. RDD, DataFrame, and Dataset Basics

## RDD

RDD stands for Resilient Distributed Dataset.

RDDs are Spark's low-level distributed data abstraction.

Example:

```python
rdd = spark.sparkContext.textFile("/data/input/customer.csv")
```

RDDs are flexible but require more manual work.

---

## DataFrame

A DataFrame is a distributed table-like structure with named columns.

Example:

```python
df = spark.read.option("header", "true").csv("/data/input/customer.csv")
```

DataFrames are usually preferred for Big Data QA because they support SQL-like operations and optimizations.

---

## Dataset

Dataset is a strongly typed API available mainly in Scala and Java.

In PySpark, you mostly work with DataFrames.

---

## RDD vs DataFrame

| Feature | RDD | DataFrame |
|---|---|---|
| Level | Low-level | Higher-level |
| Structure | Unstructured objects | Named columns |
| Optimization | Less automatic | Catalyst optimizer |
| Common QA use | Rare | Very common |
| SQL support | No direct SQL | Yes |

---

# 17. Spark Transformations and Actions

Spark uses lazy evaluation.

This means transformations are not executed immediately. They are only executed when an action is called.

---

## Transformations

Transformations create a new DataFrame/RDD.

Examples:

```python
df_filtered = df.filter(df.amount > 0)
df_selected = df.select("transaction_id", "amount")
df_grouped = df.groupBy("currency_code").count()
```

Common transformations:

| Transformation | Purpose |
|---|---|
| `select` | Select columns |
| `filter` / `where` | Filter rows |
| `withColumn` | Add/modify column |
| `drop` | Drop column |
| `groupBy` | Group data |
| `join` | Join DataFrames |
| `distinct` | Remove duplicates |
| `orderBy` | Sort data |
| `repartition` | Increase/change partitions |
| `coalesce` | Reduce partitions |

---

## Actions

Actions trigger execution.

Examples:

```python
df.count()
df.show()
df.collect()
df.write.parquet("/output/path")
```

Common actions:

| Action | Purpose |
|---|---|
| `count()` | Count rows |
| `show()` | Display sample |
| `collect()` | Bring data to driver |
| `take()` | Return limited rows |
| `write` | Save output |
| `first()` | Get first row |

---

## QA warning about collect

Avoid this on large datasets:

```python
df.collect()
```

Why?

It brings all data to the driver and can crash the job.

Better:

```python
df.limit(10).show()
```

---

# 18. Spark SQL

Spark SQL allows SQL queries on DataFrames and tables.

---

## Create temp view

```python
df.createOrReplaceTempView("transactions")
```

---

## Run SQL

```python
result = spark.sql("""
    SELECT currency_code, COUNT(*) AS record_count, SUM(amount) AS total_amount
    FROM transactions
    GROUP BY currency_code
""")

result.show()
```

---

## Read Hive table with Spark SQL

```python
df = spark.sql("""
    SELECT *
    FROM database_name.transaction_fact
    WHERE business_date = '2026-05-25'
""")
```

---

## Count validation

```python
spark.sql("""
    SELECT COUNT(*) AS record_count
    FROM transaction_fact
    WHERE business_date = '2026-05-25'
""").show()
```

---

## Duplicate validation

```python
spark.sql("""
    SELECT transaction_id, COUNT(*) AS duplicate_count
    FROM transaction_fact
    WHERE business_date = '2026-05-25'
    GROUP BY transaction_id
    HAVING COUNT(*) > 1
""").show()
```

---

# 19. PySpark for Big Data QA

PySpark is very useful for automating data checks.

---

## Start Spark session

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("BigDataQAValidation") \
    .enableHiveSupport() \
    .getOrCreate()
```

---

## Read CSV file

```python
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/data/input/transactions.csv")
```

---

## Read Parquet file

```python
df = spark.read.parquet("/data/transactions/business_date=2026-05-25/")
```

---

## Read Hive table

```python
df = spark.table("finance.transaction_fact")
```

---

## Show schema

```python
df.printSchema()
```

---

## Record count

```python
record_count = df.count()
print(f"Record count: {record_count}")
```

---

## Null count

```python
from pyspark.sql.functions import col, sum as spark_sum, when

df.select(
    spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id_count")
).show()
```

---

## Duplicate check

```python
duplicates = df.groupBy("transaction_id") \
    .count() \
    .filter(col("count") > 1)

duplicates.show()
```

---

## Invalid currency check

```python
valid_currencies = ["CAD", "USD", "EUR", "GBP"]

invalid_currency_df = df.filter(~col("currency_code").isin(valid_currencies))

invalid_currency_df.show()
```

---

## Amount reconciliation

```python
from pyspark.sql.functions import count, sum as spark_sum

df.groupBy("currency_code") \
  .agg(
      count("*").alias("record_count"),
      spark_sum("amount").alias("total_amount")
  ) \
  .show()
```

---

## Source-to-target count comparison

```python
source_df = spark.table("finance.source_transaction") \
    .filter(col("business_date") == "2026-05-25")

target_df = spark.table("finance.transaction_fact") \
    .filter(col("business_date") == "2026-05-25")

source_count = source_df.count()
target_count = target_df.count()

if source_count == target_count:
    print("PASS: Source and target counts match")
else:
    print(f"FAIL: Source count={source_count}, Target count={target_count}")
```

---

## Missing records in target

```python
missing_in_target = source_df.join(
    target_df,
    on="transaction_id",
    how="left_anti"
)

missing_in_target.show()
```

---

## Mismatched amount records

```python
mismatches = source_df.alias("s") \
    .join(target_df.alias("t"), on="transaction_id", how="inner") \
    .filter(col("s.amount") != col("t.amount")) \
    .select(
        col("transaction_id"),
        col("s.amount").alias("source_amount"),
        col("t.amount").alias("target_amount")
    )

mismatches.show()
```

---

## Save validation results

```python
mismatches.write \
    .mode("overwrite") \
    .parquet("/qa/results/amount_mismatches/business_date=2026-05-25")
```

---

# 20. Source-to-Target Testing in Big Data

Source-to-target testing checks whether data moved correctly from source to target.

---

## Common source-to-target checks

| Check | Purpose |
|---|---|
| Count check | Ensure row counts match |
| Sum check | Ensure numeric totals reconcile |
| Min/max check | Detect value range issues |
| Duplicate check | Ensure keys are unique |
| Null check | Validate mandatory fields |
| Field mapping check | Ensure transformations are correct |
| Missing record check | Find records not loaded |
| Extra record check | Find unexpected target records |
| Reject check | Ensure invalid records were rejected correctly |
| Audit check | Validate ETL metadata |

---

## SQL example

```sql
SELECT 'source' AS dataset, COUNT(*) AS record_count, SUM(amount) AS total_amount
FROM source_transaction
WHERE business_date = '2026-05-25'

UNION ALL

SELECT 'target' AS dataset, COUNT(*) AS record_count, SUM(amount) AS total_amount
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## PySpark example

```python
source_summary = source_df.groupBy("currency_code").agg(
    count("*").alias("source_count"),
    spark_sum("amount").alias("source_amount")
)

target_summary = target_df.groupBy("currency_code").agg(
    count("*").alias("target_count"),
    spark_sum("amount").alias("target_amount")
)

comparison = source_summary.join(
    target_summary,
    on="currency_code",
    how="full_outer"
)

comparison.show()
```

---

## Strong interview explanation

> I start with high-level count and amount reconciliation. If that fails, I compare by batch, date, currency, transaction type, and source system. Then I perform row-level checks using joins or anti-joins to identify missing or mismatched records. I also validate reject tables and audit tables to ensure the pipeline handled invalid records correctly.

---

# 21. Data Quality Checks

Data quality checks make sure data is accurate, complete, consistent, and usable.

---

## Core checks

| Check | Example |
|---|---|
| Completeness | Mandatory columns are not null |
| Uniqueness | Primary keys have no duplicates |
| Validity | Currency code is valid |
| Consistency | Source and target values match |
| Accuracy | Business rule result is correct |
| Timeliness | File arrived on time |
| Integrity | Foreign keys exist |
| Reconciliation | Counts and totals match |

---

## Completeness check

```sql
SELECT COUNT(*) AS missing_customer_count
FROM transaction_fact
WHERE customer_id IS NULL
  AND business_date = '2026-05-25';
```

---

## Uniqueness check

```sql
SELECT transaction_id, COUNT(*) AS duplicate_count
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Validity check

```sql
SELECT currency_code, COUNT(*) AS invalid_count
FROM transaction_fact
WHERE business_date = '2026-05-25'
  AND currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP')
GROUP BY currency_code;
```

---

## Referential integrity check

```sql
SELECT t.transaction_id, t.customer_id
FROM transaction_fact t
LEFT JOIN customer_dim c
    ON t.customer_id = c.customer_id
WHERE t.business_date = '2026-05-25'
  AND c.customer_id IS NULL;
```

---

## Business rule check

Rule: Negative amount is allowed only for reversal transactions.

```sql
SELECT *
FROM transaction_fact
WHERE business_date = '2026-05-25'
  AND amount < 0
  AND transaction_type <> 'REVERSAL';
```

---

# 22. Batch Pipeline Testing

A batch pipeline processes data on a schedule, such as hourly, daily, or monthly.

---

## Typical batch pipeline flow

```text
Source System
     ↓
Landing Zone / Raw HDFS Path
     ↓
Staging Table
     ↓
Transformation Job
     ↓
Target Hive Table / Data Lake Table
     ↓
Downstream Reports / Analytics
```

---

## QA checks at each stage

| Stage | QA Checks |
|---|---|
| Landing | File exists, file size, file count, naming convention |
| Raw/Staging | Schema, count, malformed records |
| Transform | Business rules, joins, derivations |
| Target | Count, duplicates, nulls, reconciliation |
| Reject | Invalid records captured correctly |
| Audit | Job status, start/end time, counts |
| Downstream | Report totals, data availability |

---

## Batch validation checklist

- Did the file arrive?
- Was it complete?
- Was the correct date/batch processed?
- Did the job run successfully?
- Did source count match staging count?
- Did staging count match target plus rejects?
- Were transformations correct?
- Were target partitions created?
- Were duplicate records introduced?
- Were sensitive fields protected?
- Did audit status show success?

---

# 23. Incremental Load and CDC Testing

Incremental loads process only new or changed records.

CDC means Change Data Capture.

---

## CDC operation types

| Code | Meaning |
|---|---|
| I | Insert |
| U | Update |
| D | Delete |

---

## Validate operation counts

```sql
SELECT operation_code, COUNT(*) AS record_count
FROM customer_cdc_stage
WHERE batch_id = 'BATCH_20260525'
GROUP BY operation_code;
```

---

## Validate inserts

```sql
SELECT s.customer_id
FROM customer_cdc_stage s
LEFT JOIN customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.operation_code = 'I'
  AND s.batch_id = 'BATCH_20260525'
  AND t.customer_id IS NULL;
```

This finds inserted records not present in target.

---

## Validate updates

```sql
SELECT
    s.customer_id,
    s.email AS source_email,
    t.email AS target_email
FROM customer_cdc_stage s
JOIN customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.operation_code = 'U'
  AND s.batch_id = 'BATCH_20260525'
  AND s.email <> t.email;
```

Depending on timing and SCD design, this may detect failed updates.

---

## Validate deletes

```sql
SELECT s.customer_id
FROM customer_cdc_stage s
JOIN customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.operation_code = 'D'
  AND s.batch_id = 'BATCH_20260525'
  AND t.is_active = 'Y';
```

This finds records that should have been deactivated but are still active.

---

# 24. File-Level Validation

File-level validation is very common in Big Data QA.

---

## File checks

| Check | Example |
|---|---|
| File exists | HDFS path exists |
| File is not empty | Size > 0 |
| Record count | `wc -l` or Hive count |
| Header check | Expected columns |
| Trailer check | Expected count/control total |
| File name check | Date/batch included |
| File size check | Within expected range |
| Duplicate file check | No repeated delivery |
| Compression check | `.gz`, `.snappy`, etc. |

---

## HDFS file exists check

```bash
hdfs dfs -test -e /landing/customer/customer_20260525.csv

if [ $? -eq 0 ]; then
    echo "PASS: File exists"
else
    echo "FAIL: File missing"
fi
```

---

## HDFS file size check

```bash
hdfs dfs -du -h /landing/customer/customer_20260525.csv
```

---

## HDFS record count check

```bash
hdfs dfs -cat /landing/customer/customer_20260525.csv | wc -l
```

---

## Header validation

```bash
hdfs dfs -cat /landing/customer/customer_20260525.csv | head -1
```

Expected:

```text
customer_id,first_name,last_name,email,status
```

---

## File count in landing folder

```bash
hdfs dfs -ls /landing/customer/ | grep customer_20260525 | wc -l
```

---

# 25. Partition-Level Validation

Partition validation is critical in Hive/Spark environments.

---

## Why partitions matter

Partitions help:

- Improve query performance
- Organize data by date/batch/region
- Avoid scanning full tables
- Make validation easier

---

## Check partition exists

```sql
SHOW PARTITIONS transaction_fact;
```

---

## Validate specific partition count

```sql
SELECT COUNT(*) AS record_count
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Validate partition amount

```sql
SELECT
    currency_code,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY currency_code;
```

---

## Find unexpected partitions

```sql
SELECT DISTINCT business_date
FROM transaction_fact
WHERE business_date > '2026-05-25';
```

---

## PySpark partition check

```python
df = spark.read.parquet("/warehouse/transaction_fact/business_date=2026-05-25")

print(df.count())
df.printSchema()
```

---

# 26. Reject and Audit Table Validation

Reject and audit tables are extremely important in enterprise QA.

---

## Reject table

A reject table stores records that failed validation.

Example columns:

```text
record_id
source_file
reject_reason
reject_timestamp
batch_id
raw_record
```

---

## Audit table

An audit table stores job-level metadata.

Example columns:

```text
job_name
batch_id
source_count
target_count
reject_count
job_status
start_time
end_time
error_message
```

---

## Validate job status

```sql
SELECT *
FROM etl_audit
WHERE batch_id = 'BATCH_20260525'
  AND job_status <> 'SUCCESS';
```

---

## Validate count reconciliation using audit table

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
    END AS validation_status
FROM etl_audit
WHERE batch_id = 'BATCH_20260525';
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

## Strong interview explanation

> I always check audit and reject tables because a target count mismatch may be valid if some records were rejected according to business rules. The correct reconciliation is usually source count equals target count plus reject count, depending on pipeline design.

---

# 27. Spark Performance Concepts

You do not need to be a Spark expert for Big Data QA, but you should know core performance concepts.

---

## Lazy evaluation

Spark does not execute transformations immediately. It builds a plan and executes it only when an action is called.

Example:

```python
filtered_df = df.filter(col("amount") > 0)
```

This does not execute yet.

```python
filtered_df.count()
```

This triggers execution.

---

## Shuffle

A shuffle happens when Spark redistributes data across partitions.

Common operations causing shuffle:

- `groupBy`
- `join`
- `distinct`
- `orderBy`
- `repartition`

Why QA should care:

> Jobs with big shuffles can be slow or fail due to memory issues. If a validation query uses large joins or aggregations, I would filter by partition and validate in smaller groups first.

---

## Cache and persist

Caching stores intermediate DataFrames for reuse.

```python
df_filtered = df.filter(col("business_date") == "2026-05-25")
df_filtered.cache()

df_filtered.count()
df_filtered.groupBy("currency_code").count().show()
```

Use cache when the same DataFrame is reused multiple times.

---

## Repartition vs coalesce

| Function | Meaning |
|---|---|
| `repartition()` | Changes partition count with shuffle |
| `coalesce()` | Reduces partition count with less shuffle |

Example:

```python
df_repartitioned = df.repartition(100)
df_coalesced = df.coalesce(10)
```

---

## Broadcast join

Broadcast join sends a small table to all executors.

Example:

```python
from pyspark.sql.functions import broadcast

result = large_df.join(
    broadcast(small_lookup_df),
    on="currency_code",
    how="left"
)
```

Use when joining a large fact table with a small reference table.

---

## Data skew

Data skew happens when some keys have much more data than others.

Example:

```text
currency_code = CAD has 90% of records
currency_code = EUR has 2% of records
```

Symptoms:

- Some tasks run much longer.
- Job appears stuck near the end.
- Large shuffle partitions.
- Executor memory errors.

QA-level explanation:

> If a Spark job is slow or stuck, I would check whether the data is skewed by grouping on join keys or partition keys and checking record distribution.

---

# 28. Hive Performance Concepts

Hive performance often depends on how the table is stored and queried.

---

## Partition pruning

Partition pruning means Hive reads only relevant partitions.

Good:

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

Bad:

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE substr(business_date, 1, 10) = '2026-05-25';
```

Using functions on partition columns can prevent efficient pruning.

---

## Columnar formats

ORC and Parquet can improve performance because they store data by column.

If a query only needs `transaction_id` and `amount`, it may avoid reading unnecessary columns.

---

## Compression

Compression reduces storage and I/O.

Common compression examples:

- Snappy
- Gzip
- Zlib

---

## Small files problem

Too many small files can slow down Hive and Spark.

Why?

- More metadata
- More file open operations
- More tasks
- Inefficient distributed processing

QA example:

```bash
hdfs dfs -ls /warehouse/transaction_fact/business_date=2026-05-25 | wc -l
```

If a partition has thousands of tiny files, it may indicate a performance issue.

---

## Explain plan

```sql
EXPLAIN
SELECT currency_code, COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY currency_code;
```

Use `EXPLAIN` to understand whether the query is scanning too much data.

---

# 29. Common Production Issues and Debugging

## Issue 1: File missing

Symptoms:

- Job did not start.
- Source count is zero.
- Audit table says input missing.

Checks:

```bash
hdfs dfs -ls /landing/customer/
hdfs dfs -test -e /landing/customer/customer_20260525.csv
```

---

## Issue 2: Empty file

Checks:

```bash
hdfs dfs -du -h /landing/customer/customer_20260525.csv
hdfs dfs -cat /landing/customer/customer_20260525.csv | wc -l
```

---

## Issue 3: Schema mismatch

Symptoms:

- Spark job fails reading file.
- Hive table returns nulls.
- Column shifts happen in CSV.
- Target has unexpected nulls.

Checks:

```python
df.printSchema()
df.show(5, truncate=False)
```

Hive:

```sql
DESCRIBE FORMATTED table_name;
```

---

## Issue 4: Duplicate records

Checks:

```sql
SELECT transaction_id, COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Issue 5: Partition not created

Checks:

```sql
SHOW PARTITIONS transaction_fact;
```

Repair metadata if needed:

```sql
MSCK REPAIR TABLE transaction_fact;
```

Use this only when appropriate for the environment.

---

## Issue 6: Spark job failed

Checks:

- Application logs
- YARN logs
- Spark UI
- Executor logs
- Driver logs
- Input path
- Permissions
- Schema
- Memory errors
- Data skew
- Bad records

Command:

```bash
yarn logs -applicationId <application_id>
```

---

## Issue 7: Count mismatch

Debug sequence:

1. Check source count.
2. Check target count.
3. Check reject count.
4. Check audit table.
5. Compare by partition/date.
6. Compare by source system.
7. Compare by transaction type/currency.
8. Use anti-join to find missing records.
9. Check duplicate keys.
10. Check job logs for filtering or rejection messages.

---

# 30. Banking Big Data QA Scenarios

## Scenario 1: Daily transaction load validation

Checks:

```sql
SELECT COUNT(*) AS source_count
FROM source_transaction
WHERE business_date = '2026-05-25';

SELECT COUNT(*) AS target_count
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

Amount reconciliation:

```sql
SELECT currency_code, SUM(amount) AS total_amount
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY currency_code;
```

---

## Scenario 2: Negative transaction amount rule

```sql
SELECT *
FROM transaction_fact
WHERE business_date = '2026-05-25'
  AND amount < 0
  AND transaction_type <> 'REVERSAL';
```

Expected result: zero rows.

---

## Scenario 3: Customer dimension current record validation

```sql
SELECT customer_id, COUNT(*) AS current_record_count
FROM customer_dim
WHERE is_current = 'Y'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

Expected result: zero rows.

---

## Scenario 4: Account reference integrity

```sql
SELECT t.transaction_id, t.account_id
FROM transaction_fact t
LEFT JOIN account_dim a
    ON t.account_id = a.account_id
WHERE t.business_date = '2026-05-25'
  AND a.account_id IS NULL;
```

Expected result: zero rows unless invalid records are allowed and rejected elsewhere.

---

## Scenario 5: Batch audit check

```sql
SELECT
    job_name,
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

## Scenario 6: Sensitive data check in logs

In banking environments, sensitive fields should not be printed in plain logs.

Example shell check:

```bash
grep -Ei "ssn|sin|account_number|card_number|password" application.log
```

If found, report it carefully according to team process.

---

# 31. Interview Questions and Model Answers

## Q1. What is Hadoop?

**Answer:**  
Hadoop is an ecosystem for distributed storage and processing of large datasets. HDFS provides distributed storage, and tools like MapReduce, Hive, and Spark process or query the data.

---

## Q2. What is HDFS?

**Answer:**  
HDFS is the Hadoop Distributed File System. It stores large files by splitting them into blocks and distributing those blocks across DataNodes. It uses replication for fault tolerance and is optimized for high-throughput access.

---

## Q3. What are NameNode and DataNode?

**Answer:**  
The NameNode manages metadata, such as file names, directories, and block locations. DataNodes store the actual data blocks.

---

## Q4. How do you check if a file exists in HDFS?

```bash
hdfs dfs -test -e /data/input/file.csv
echo $?
```

If the return code is `0`, the file exists.

---

## Q5. How do you count records in an HDFS file?

```bash
hdfs dfs -cat /data/input/file.csv | wc -l
```

If the file has a header:

```bash
hdfs dfs -cat /data/input/file.csv | tail -n +2 | wc -l
```

---

## Q6. What is Hive?

**Answer:**  
Hive is a data warehouse layer that allows SQL-style queries on data stored in Hadoop or distributed storage. It is commonly used for batch analytics and data validation.

---

## Q7. Managed table vs external table?

**Answer:**  
A managed table is owned by Hive, and Hive manages both metadata and data. An external table points to data stored outside Hive ownership. Dropping an external table usually removes only metadata, while dropping a managed table may remove the data depending on configuration.

---

## Q8. What is partitioning in Hive?

**Answer:**  
Partitioning stores data in separate directories based on column values, such as business date. It improves query performance because queries can scan only relevant partitions.

---

## Q9. What is Spark?

**Answer:**  
Spark is a distributed processing engine for large-scale data. It supports APIs in Python, Scala, Java, and R and includes Spark SQL for structured data processing.

---

## Q10. What is lazy evaluation in Spark?

**Answer:**  
Spark does not execute transformations immediately. It builds a logical execution plan and runs it only when an action like `count()`, `show()`, or `write()` is called.

---

## Q11. Transformation vs action in Spark?

**Answer:**  
A transformation creates a new DataFrame or RDD, such as `filter`, `select`, or `join`. An action triggers execution, such as `count`, `show`, or `write`.

---

## Q12. How do you find duplicates using Spark?

```python
duplicates = df.groupBy("transaction_id").count().filter("count > 1")
duplicates.show()
```

---

## Q13. How would you validate source and target data in Spark?

**Answer:**  
I would filter both source and target by batch or business date, compare counts and aggregate totals, check duplicates and nulls, and then use joins or anti-joins to identify missing or mismatched records.

---

## Q14. What is a shuffle in Spark?

**Answer:**  
A shuffle is data redistribution across partitions. It often happens during joins, groupBy, distinct, and orderBy. Shuffles can be expensive and may cause performance issues on large datasets.

---

## Q15. What is data skew?

**Answer:**  
Data skew occurs when data is unevenly distributed across partitions or keys. Some tasks process much more data than others, causing slow jobs or failures.

---

## Q16. How would you debug a failed Spark job?

**Answer:**  
I would check the job logs, Spark UI or YARN logs, input path, schema, permissions, partition filters, bad records, executor memory errors, and data skew. I would also validate whether the failure happened during read, transformation, shuffle, or write.

---

## Q17. How would you test a Big Data pipeline?

**Answer:**  
I would test it in stages: file arrival, schema, staging load, transformation logic, target load, partition creation, source-to-target reconciliation, reject validation, audit validation, and downstream checks. I would also review logs and ensure sensitive data is not exposed.

---

## Q18. How do you validate a Hive partition?

```sql
SHOW PARTITIONS transaction_fact;

SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Q19. What is the small files problem?

**Answer:**  
The small files problem happens when a table or partition contains too many small files. It increases metadata overhead and creates many small tasks, which can slow down Hive and Spark jobs.

---

## Q20. Why are Parquet and ORC commonly used?

**Answer:**  
Parquet and ORC are columnar formats. They are efficient for analytics because queries can read only required columns and benefit from compression.

---

# 32. Hands-On Practice Tasks

## Task 1: Check file exists in HDFS

```bash
hdfs dfs -test -e /landing/transactions/txn_20260525.csv
echo $?
```

---

## Task 2: Count records in an HDFS file

```bash
hdfs dfs -cat /landing/transactions/txn_20260525.csv | tail -n +2 | wc -l
```

---

## Task 3: Check Hive table count

```sql
SELECT COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25';
```

---

## Task 4: Find duplicates in Hive

```sql
SELECT transaction_id, COUNT(*)
FROM transaction_fact
WHERE business_date = '2026-05-25'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Task 5: Read Parquet file in PySpark

```python
df = spark.read.parquet("/warehouse/transaction_fact/business_date=2026-05-25")
df.printSchema()
df.count()
```

---

## Task 6: Compare source and target counts in PySpark

```python
source_count = source_df.count()
target_count = target_df.count()

print(source_count, target_count)
```

---

## Task 7: Find missing records in PySpark

```python
missing = source_df.join(target_df, on="transaction_id", how="left_anti")
missing.show()
```

---

## Task 8: Aggregate reconciliation in Spark

```python
source_summary = source_df.groupBy("currency_code").agg(
    count("*").alias("source_count"),
    spark_sum("amount").alias("source_amount")
)

target_summary = target_df.groupBy("currency_code").agg(
    count("*").alias("target_count"),
    spark_sum("amount").alias("target_amount")
)

source_summary.join(target_summary, "currency_code", "full_outer").show()
```

---

## Task 9: Check reject table

```sql
SELECT reject_reason, COUNT(*)
FROM reject_records
WHERE batch_id = 'BATCH_20260525'
GROUP BY reject_reason;
```

---

## Task 10: Check audit table

```sql
SELECT *
FROM etl_audit
WHERE batch_id = 'BATCH_20260525';
```

---

# 33. 10-Day Preparation Plan

## Day 1: Big Data fundamentals

Study:

- What Big Data means
- Hadoop ecosystem
- HDFS
- Hive
- Spark
- Batch vs streaming

Goal: Explain the ecosystem clearly.

---

## Day 2: HDFS commands

Practice:

```bash
hdfs dfs -ls
hdfs dfs -cat
hdfs dfs -head
hdfs dfs -tail
hdfs dfs -du -h
hdfs dfs -count
hdfs dfs -test -e
hdfs dfs -put
hdfs dfs -get
```

Goal: Validate files confidently.

---

## Day 3: Hive basics

Practice:

- `CREATE TABLE`
- `CREATE EXTERNAL TABLE`
- `DESCRIBE`
- `SHOW TABLES`
- `SHOW PARTITIONS`
- `SELECT`
- `COUNT`
- `GROUP BY`

Goal: Query and inspect Hive tables.

---

## Day 4: Hive QA validation

Practice:

- Count checks
- Duplicate checks
- Null checks
- Invalid value checks
- Source-to-target reconciliation
- Partition validation

Goal: Write QA SQL quickly.

---

## Day 5: Spark concepts

Study:

- Driver
- Executor
- Task
- Stage
- Job
- DAG
- Lazy evaluation
- Transformations/actions

Goal: Explain Spark architecture at interview level.

---

## Day 6: PySpark basics

Practice:

- Create SparkSession
- Read CSV
- Read Parquet
- Print schema
- Count rows
- Filter data
- Group data
- Join DataFrames

Goal: Write basic PySpark validation scripts.

---

## Day 7: Source-to-target testing

Practice:

- Count reconciliation
- Sum reconciliation
- Missing record checks
- Mismatch checks
- Anti-joins
- Full outer comparisons

Goal: Explain a full validation approach.

---

## Day 8: Performance concepts

Study:

- Partitions
- Shuffles
- Data skew
- Broadcast joins
- Cache/persist
- Small files problem
- Columnar file formats

Goal: Discuss performance at QA level.

---

## Day 9: Production debugging

Practice explaining:

- File missing
- Empty file
- Schema mismatch
- Count mismatch
- Partition missing
- Spark job failure
- Hive query slow
- Duplicates introduced

Goal: Give structured debugging answers.

---

## Day 10: Mock interview

Practice answering:

- What is HDFS?
- What is Hive?
- What is Spark?
- How do you validate a pipeline?
- How do you debug a count mismatch?
- How do you find duplicates?
- What is lazy evaluation?
- What is data skew?
- What is partition pruning?
- How do you validate audit and reject tables?

Goal: Sound confident and practical.

---

# 34. Final Cheat Sheet

## HDFS commands

```bash
hdfs dfs -ls /path
hdfs dfs -cat /path/file.csv
hdfs dfs -head /path/file.csv
hdfs dfs -tail /path/file.csv
hdfs dfs -du -h /path
hdfs dfs -count /path
hdfs dfs -test -e /path/file.csv
hdfs dfs -put local.csv /hdfs/path
hdfs dfs -get /hdfs/path/file.csv .
hdfs dfs -rm /path/file.csv
```

---

## YARN commands

```bash
yarn application -list
yarn application -status <application_id>
yarn logs -applicationId <application_id>
```

---

## Hive commands

```sql
SHOW DATABASES;
SHOW TABLES;
DESCRIBE table_name;
DESCRIBE FORMATTED table_name;
SHOW PARTITIONS table_name;
SELECT COUNT(*) FROM table_name;
```

---

## Hive QA SQL

```sql
-- Count
SELECT COUNT(*) FROM transaction_fact WHERE business_date = '2026-05-25';

-- Duplicate
SELECT transaction_id, COUNT(*)
FROM transaction_fact
GROUP BY transaction_id
HAVING COUNT(*) > 1;

-- Null
SELECT COUNT(*)
FROM transaction_fact
WHERE customer_id IS NULL;

-- Invalid domain
SELECT currency_code, COUNT(*)
FROM transaction_fact
WHERE currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP')
GROUP BY currency_code;

-- Source-to-target
SELECT 'source', COUNT(*) FROM source_transaction
UNION ALL
SELECT 'target', COUNT(*) FROM transaction_fact;
```

---

## PySpark essentials

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, when

spark = SparkSession.builder.appName("QA").enableHiveSupport().getOrCreate()

df = spark.read.option("header", "true").csv("/data/input/file.csv")
df.printSchema()
df.count()
df.show(5)

df.groupBy("currency_code").agg(
    count("*").alias("record_count"),
    spark_sum("amount").alias("total_amount")
).show()

duplicates = df.groupBy("transaction_id").count().filter(col("count") > 1)
duplicates.show()
```

---

## Key interview phrases

Use these phrases:

- Distributed storage
- Distributed processing
- HDFS blocks
- NameNode and DataNode
- Hive external table
- Partition pruning
- Spark driver and executors
- Lazy evaluation
- Transformations and actions
- Shuffle
- Data skew
- Source-to-target validation
- Count reconciliation
- Amount reconciliation
- Audit table validation
- Reject table validation
- Partition-level validation
- Production debugging
- Data quality checks
- Sensitive data protection

---

## Strong closing answer

If asked, **“How would you validate a Big Data pipeline?”**, say:

> I would validate it in layers. First, I would check file arrival in HDFS, file size, naming convention, and record count. Then I would validate the staging table schema, nulls, duplicates, and invalid values. After transformation, I would compare source and target counts, sums, and key business fields using Hive SQL or Spark. I would also validate target partitions, audit table counts, reject records, and job logs. If there is a mismatch, I would narrow it down by batch, date, source system, currency, transaction type, and finally use anti-joins or row-level comparisons to identify the exact records.

---

# 35. Reference Links

Use these for deeper study:

- RBC Careers: Quality Engineer and Data Engineer postings often mention Unix/Linux, SQL, Python, Big Data/Hadoop, Spark, large-scale databases, automated testing, and data pipeline work.
- Apache Hadoop FileSystem Shell: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html
- Apache Hadoop HDFS Commands Guide: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HDFSCommands.html
- Apache Hadoop HDFS Architecture: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Apache Spark SQL, DataFrames and Datasets Guide: https://spark.apache.org/docs/latest/sql-programming-guide.html
- Apache Hive Language Manual: https://hive.apache.org/docs/latest/language/
- Apache Hive DDL Manual: https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL
