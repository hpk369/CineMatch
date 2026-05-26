# SQL Interview Guide for Big Data QA / Quality Engineer Role at RBC

**Target role:** Big Data QA, Quality Engineer, Data QA Analyst, ETL QA, Data Validation Engineer  
**Target company context:** RBC / banking / financial services / enterprise data platforms  
**Main focus:** SQL for source-to-target testing, reconciliation, data quality validation, ETL testing, Hadoop/Hive/Spark SQL, and production-style debugging.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)  
2. [What SQL Skills RBC-Style Big Data QA Interviews Test](#2-what-sql-skills-rbc-style-big-data-qa-interviews-test)  
3. [Sample Tables Used Throughout the Guide](#3-sample-tables-used-throughout-the-guide)  
4. [SQL Basics You Must Know](#4-sql-basics-you-must-know)  
5. [Filtering, Sorting, and Conditional Logic](#5-filtering-sorting-and-conditional-logic)  
6. [Aggregations for QA Validation](#6-aggregations-for-qa-validation)  
7. [Joins for Source-to-Target Testing](#7-joins-for-source-to-target-testing)  
8. [Duplicate Checks](#8-duplicate-checks)  
9. [Null, Blank, and Invalid Value Checks](#9-null-blank-and-invalid-value-checks)  
10. [Date and Timestamp Validation](#10-date-and-timestamp-validation)  
11. [Data Type and Format Validation](#11-data-type-and-format-validation)  
12. [Source-to-Target Reconciliation](#12-source-to-target-reconciliation)  
13. [Minus / Except / Anti-Join Testing](#13-minus--except--anti-join-testing)  
14. [Window Functions](#14-window-functions)  
15. [CTEs and Subqueries](#15-ctes-and-subqueries)  
16. [Case Statements and Business Rule Testing](#16-case-statements-and-business-rule-testing)  
17. [Incremental Load and CDC Testing](#17-incremental-load-and-cdc-testing)  
18. [SCD Type 1 and Type 2 Testing](#18-scd-type-1-and-type-2-testing)  
19. [Big Data SQL: Hive and Spark SQL](#19-big-data-sql-hive-and-spark-sql)  
20. [SQL Performance Concepts for QA](#20-sql-performance-concepts-for-qa)  
21. [Banking and Financial Data Validation Scenarios](#21-banking-and-financial-data-validation-scenarios)  
22. [SQL Interview Questions and Model Answers](#22-sql-interview-questions-and-model-answers)  
23. [Hands-On SQL Practice Exercises](#23-hands-on-sql-practice-exercises)  
24. [7-Day SQL Preparation Plan](#24-7-day-sql-preparation-plan)  
25. [Final Cheat Sheet](#25-final-cheat-sheet)  
26. [Reference Links](#26-reference-links)

---

# 1. How to Use This Guide

This guide is designed for interview preparation for a **Big Data QA / Quality Engineer** role where SQL is used to validate data pipelines.

Do not study SQL only as a database developer. For this type of role, focus on how SQL helps you answer QA questions such as:

- Did all source records load into the target?
- Are there duplicate primary keys?
- Are mandatory fields null?
- Are business rules applied correctly?
- Are rejected records expected?
- Are amounts, balances, and counts reconciled?
- Are incremental records loaded correctly?
- Are old records updated correctly?
- Are there records missing between source and target?
- Are Hive/Spark SQL queries producing the expected output?

A good SQL answer in a Big Data QA interview should usually include:

1. What you are validating.
2. Which source and target tables/files you are comparing.
3. The SQL query you would use.
4. How you interpret pass/fail.
5. What you would check next if the validation fails.

---

# 2. What SQL Skills RBC-Style Big Data QA Interviews Test

For an RBC-style Big Data QA or Quality Engineer role, SQL is usually tested in the context of:

- Data warehouse testing
- ETL testing
- Source-to-target validation
- Hadoop/Hive/Spark-based data validation
- Batch data pipeline testing
- Data reconciliation
- Automated QA checks
- Banking data controls
- Production incident analysis

You should be able to write SQL queries for:

| Skill Area | Why It Matters |
|---|---|
| `SELECT`, `WHERE`, `ORDER BY` | Basic data inspection |
| `GROUP BY`, `HAVING` | Count, sum, duplicate, and reconciliation checks |
| `INNER JOIN` | Matching source and target records |
| `LEFT JOIN` | Finding missing target records |
| `FULL OUTER JOIN` | Finding mismatches both ways |
| `CASE WHEN` | Business rule validation |
| CTEs | Readable validation queries |
| Window functions | Duplicate ranking, latest record selection, SCD testing |
| `EXCEPT` / `MINUS` | Row-level mismatch testing |
| Date functions | Batch and incremental load testing |
| String functions | Format and data cleansing checks |
| Performance basics | Working with large datasets |

---

# 3. Sample Tables Used Throughout the Guide

The examples in this guide use banking-style tables.

## Source customer table

```sql
source_customer
---------------
customer_id
first_name
last_name
email
phone_number
country_code
customer_status
created_date
updated_timestamp
batch_id
```

## Target customer dimension table

```sql
target_customer_dim
-------------------
customer_key
customer_id
full_name
email
phone_number
country_code
customer_status
effective_start_date
effective_end_date
is_current
created_date
updated_timestamp
batch_id
```

## Source transaction table

```sql
source_transaction
------------------
transaction_id
account_id
customer_id
transaction_date
transaction_type
currency_code
amount
batch_id
```

## Target transaction fact table

```sql
target_transaction_fact
-----------------------
transaction_id
account_id
customer_id
transaction_date
transaction_type
currency_code
amount
load_timestamp
batch_id
```

## Reject table

```sql
reject_records
--------------
record_id
source_table
reject_reason
reject_timestamp
batch_id
```

---

# 4. SQL Basics You Must Know

## Basic SELECT

```sql
SELECT *
FROM source_customer;
```

In interviews, avoid always using `SELECT *`. It is okay for quick inspection, but for validation you should select only the columns needed.

Better:

```sql
SELECT customer_id, email, customer_status
FROM source_customer;
```

---

## Select distinct values

```sql
SELECT DISTINCT customer_status
FROM source_customer;
```

Use this to check unexpected values.

Example:

```sql
SELECT DISTINCT currency_code
FROM source_transaction;
```

If a banking system supports only `CAD`, `USD`, `EUR`, and `GBP`, then any other currency code may need investigation.

---

## Count records

```sql
SELECT COUNT(*) AS total_records
FROM source_customer;
```

This is the most basic validation query.

---

## Count non-null values

```sql
SELECT COUNT(email) AS non_null_email_count
FROM source_customer;
```

`COUNT(column_name)` counts only non-null values.  
`COUNT(*)` counts all rows.

---

## Check a sample of records

```sql
SELECT *
FROM source_transaction
FETCH FIRST 10 ROWS ONLY;
```

In SQL Server:

```sql
SELECT TOP 10 *
FROM source_transaction;
```

In Hive/Spark/PostgreSQL/MySQL:

```sql
SELECT *
FROM source_transaction
LIMIT 10;
```

---

# 5. Filtering, Sorting, and Conditional Logic

## WHERE clause

```sql
SELECT *
FROM source_transaction
WHERE amount > 10000;
```

Use this for business rule testing.

---

## Multiple conditions

```sql
SELECT *
FROM source_transaction
WHERE amount > 10000
  AND currency_code = 'CAD';
```

---

## OR condition

```sql
SELECT *
FROM source_transaction
WHERE transaction_type = 'REVERSAL'
   OR amount < 0;
```

---

## IN condition

```sql
SELECT *
FROM source_transaction
WHERE currency_code IN ('CAD', 'USD', 'EUR', 'GBP');
```

---

## NOT IN condition

```sql
SELECT *
FROM source_transaction
WHERE currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP');
```

This is useful for invalid domain value checks.

---

## BETWEEN

```sql
SELECT *
FROM source_transaction
WHERE transaction_date BETWEEN DATE '2026-05-01' AND DATE '2026-05-31';
```

---

## LIKE

```sql
SELECT *
FROM source_customer
WHERE email LIKE '%@%';
```

Basic email validation can start with `LIKE`, but real email validation may require stronger rules or application-level validation.

---

## ORDER BY

```sql
SELECT transaction_id, amount, transaction_date
FROM source_transaction
ORDER BY amount DESC;
```

Useful for identifying unusually large values.

---

# 6. Aggregations for QA Validation

Aggregations are extremely important in Big Data QA.

## Count by batch

```sql
SELECT batch_id, COUNT(*) AS record_count
FROM source_transaction
GROUP BY batch_id
ORDER BY batch_id;
```

---

## Count by status

```sql
SELECT customer_status, COUNT(*) AS record_count
FROM source_customer
GROUP BY customer_status
ORDER BY record_count DESC;
```

This helps identify unexpected distribution changes.

---

## Sum validation

```sql
SELECT 
    batch_id,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM source_transaction
GROUP BY batch_id;
```

In banking QA, sum reconciliation is often just as important as count reconciliation.

---

## Min, max, and average

```sql
SELECT
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    AVG(amount) AS avg_amount
FROM source_transaction;
```

Use this to detect outliers or invalid values.

---

## HAVING clause

`WHERE` filters rows before aggregation.  
`HAVING` filters groups after aggregation.

Find duplicate transaction IDs:

```sql
SELECT transaction_id, COUNT(*) AS duplicate_count
FROM source_transaction
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

# 7. Joins for Source-to-Target Testing

Joins are one of the most important SQL topics for Big Data QA interviews.

## INNER JOIN

Use `INNER JOIN` when you want matching records in both source and target.

```sql
SELECT 
    s.customer_id,
    s.email AS source_email,
    t.email AS target_email
FROM source_customer s
INNER JOIN target_customer_dim t
    ON s.customer_id = t.customer_id;
```

---

## LEFT JOIN

Use `LEFT JOIN` to find source records missing in the target.

```sql
SELECT s.customer_id
FROM source_customer s
LEFT JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE t.customer_id IS NULL;
```

Interview explanation:

> This query returns records that exist in the source but did not load into the target.

---

## RIGHT JOIN

Use `RIGHT JOIN` to find target records that do not exist in source.

```sql
SELECT t.customer_id
FROM source_customer s
RIGHT JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.customer_id IS NULL;
```

Many teams prefer rewriting this as a `LEFT JOIN` by switching table order.

```sql
SELECT t.customer_id
FROM target_customer_dim t
LEFT JOIN source_customer s
    ON t.customer_id = s.customer_id
WHERE s.customer_id IS NULL;
```

---

## FULL OUTER JOIN

Use `FULL OUTER JOIN` to identify records missing on either side.

```sql
SELECT
    COALESCE(s.customer_id, t.customer_id) AS customer_id,
    CASE
        WHEN s.customer_id IS NULL THEN 'Missing in source'
        WHEN t.customer_id IS NULL THEN 'Missing in target'
        ELSE 'Present in both'
    END AS comparison_status
FROM source_customer s
FULL OUTER JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.customer_id IS NULL
   OR t.customer_id IS NULL;
```

---

## Join interview question

**Question:** How do you find records loaded in source but missing in target?

**Answer:**

```sql
SELECT s.*
FROM source_table s
LEFT JOIN target_table t
    ON s.primary_key = t.primary_key
WHERE t.primary_key IS NULL;
```

Explain:

> I would use a left join from source to target and filter where the target key is null. This tells me which source records did not load into target.

---

# 8. Duplicate Checks

Duplicate checks are common in QA interviews.

## Find duplicate customer IDs

```sql
SELECT customer_id, COUNT(*) AS duplicate_count
FROM source_customer
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Find duplicate transaction IDs

```sql
SELECT transaction_id, COUNT(*) AS duplicate_count
FROM target_transaction_fact
GROUP BY transaction_id
HAVING COUNT(*) > 1;
```

---

## Count total duplicate keys

```sql
SELECT COUNT(*) AS duplicate_key_count
FROM (
    SELECT transaction_id
    FROM target_transaction_fact
    GROUP BY transaction_id
    HAVING COUNT(*) > 1
) d;
```

---

## Find full duplicate rows

```sql
SELECT
    customer_id,
    email,
    phone_number,
    customer_status,
    COUNT(*) AS row_count
FROM source_customer
GROUP BY customer_id, email, phone_number, customer_status
HAVING COUNT(*) > 1;
```

---

## Use ROW_NUMBER to identify duplicates

```sql
SELECT *
FROM (
    SELECT
        customer_id,
        email,
        updated_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM source_customer
) x
WHERE rn > 1;
```

This returns duplicate customer records except the latest one.

---

# 9. Null, Blank, and Invalid Value Checks

## Check null customer IDs

```sql
SELECT COUNT(*) AS null_customer_id_count
FROM source_customer
WHERE customer_id IS NULL;
```

---

## Check blank strings

```sql
SELECT COUNT(*) AS blank_email_count
FROM source_customer
WHERE TRIM(email) = '';
```

---

## Check null or blank together

```sql
SELECT *
FROM source_customer
WHERE email IS NULL
   OR TRIM(email) = '';
```

---

## Mandatory column validation

```sql
SELECT
    SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id_count,
    SUM(CASE WHEN email IS NULL OR TRIM(email) = '' THEN 1 ELSE 0 END) AS invalid_email_count,
    SUM(CASE WHEN customer_status IS NULL THEN 1 ELSE 0 END) AS null_status_count
FROM source_customer;
```

This is a strong QA query because it returns multiple validation results in one output.

---

## Invalid status values

```sql
SELECT customer_status, COUNT(*) AS record_count
FROM source_customer
WHERE customer_status NOT IN ('ACTIVE', 'INACTIVE', 'CLOSED')
GROUP BY customer_status;
```

---

## Invalid currency values

```sql
SELECT currency_code, COUNT(*) AS record_count
FROM source_transaction
WHERE currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP')
GROUP BY currency_code;
```

---

# 10. Date and Timestamp Validation

Date validation is very important in batch pipelines.

## Future-dated transactions

```sql
SELECT *
FROM source_transaction
WHERE transaction_date > CURRENT_DATE;
```

---

## Missing transaction dates

```sql
SELECT *
FROM source_transaction
WHERE transaction_date IS NULL;
```

---

## Records loaded for a specific batch date

```sql
SELECT COUNT(*) AS record_count
FROM target_transaction_fact
WHERE CAST(load_timestamp AS DATE) = DATE '2026-05-25';
```

Syntax may vary by database.

---

## Check processing date against business date

```sql
SELECT *
FROM source_transaction
WHERE transaction_date > CAST(load_timestamp AS DATE);
```

---

## Check date range by batch

```sql
SELECT
    batch_id,
    MIN(transaction_date) AS min_transaction_date,
    MAX(transaction_date) AS max_transaction_date,
    COUNT(*) AS record_count
FROM target_transaction_fact
GROUP BY batch_id
ORDER BY batch_id;
```

This helps detect wrong-date data accidentally loaded into a batch.

---

# 11. Data Type and Format Validation

## Numeric validation

In some databases, values may arrive as strings in staging tables.

Example: amount is stored as text in a raw staging table.

```sql
SELECT *
FROM raw_transaction_stage
WHERE amount_text IS NULL
   OR TRIM(amount_text) = '';
```

For database-specific regex support:

```sql
SELECT *
FROM raw_transaction_stage
WHERE amount_text NOT RLIKE '^-?[0-9]+(\\.[0-9]{1,2})?$';
```

`RLIKE` is commonly used in Hive/Spark SQL.

---

## Email format basic check

```sql
SELECT *
FROM source_customer
WHERE email NOT LIKE '%@%.%';
```

This is basic and not perfect, but it is often acceptable for an interview-level data quality example.

---

## Phone number length check

```sql
SELECT *
FROM source_customer
WHERE LENGTH(REGEXP_REPLACE(phone_number, '[^0-9]', '')) <> 10;
```

Syntax varies by database.

---

## Country code length check

```sql
SELECT *
FROM source_customer
WHERE LENGTH(country_code) <> 2;
```

---

# 12. Source-to-Target Reconciliation

Source-to-target testing is one of the most important areas for Big Data QA.

## Count reconciliation

```sql
SELECT 'source' AS dataset_name, COUNT(*) AS record_count
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

UNION ALL

SELECT 'target' AS dataset_name, COUNT(*) AS record_count
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

Expected result:

| dataset_name | record_count |
|---|---:|
| source | 1000000 |
| target | 1000000 |

If counts match, the count check passes. If not, continue with missing-record checks.

---

## Sum reconciliation

```sql
SELECT 'source' AS dataset_name, SUM(amount) AS total_amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

UNION ALL

SELECT 'target' AS dataset_name, SUM(amount) AS total_amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

---

## Count and sum together

```sql
SELECT
    'source' AS dataset_name,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

UNION ALL

SELECT
    'target' AS dataset_name,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

---

## Group-level reconciliation

```sql
SELECT
    transaction_date,
    currency_code,
    COUNT(*) AS source_count,
    SUM(amount) AS source_amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'
GROUP BY transaction_date, currency_code;
```

Compare with target:

```sql
SELECT
    transaction_date,
    currency_code,
    COUNT(*) AS target_count,
    SUM(amount) AS target_amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525'
GROUP BY transaction_date, currency_code;
```

A stronger single-query comparison:

```sql
WITH src AS (
    SELECT
        transaction_date,
        currency_code,
        COUNT(*) AS source_count,
        SUM(amount) AS source_amount
    FROM source_transaction
    WHERE batch_id = 'BATCH_20260525'
    GROUP BY transaction_date, currency_code
),
tgt AS (
    SELECT
        transaction_date,
        currency_code,
        COUNT(*) AS target_count,
        SUM(amount) AS target_amount
    FROM target_transaction_fact
    WHERE batch_id = 'BATCH_20260525'
    GROUP BY transaction_date, currency_code
)
SELECT
    COALESCE(src.transaction_date, tgt.transaction_date) AS transaction_date,
    COALESCE(src.currency_code, tgt.currency_code) AS currency_code,
    src.source_count,
    tgt.target_count,
    src.source_amount,
    tgt.target_amount,
    CASE
        WHEN src.source_count = tgt.target_count
         AND src.source_amount = tgt.target_amount
        THEN 'PASS'
        ELSE 'FAIL'
    END AS validation_status
FROM src
FULL OUTER JOIN tgt
    ON src.transaction_date = tgt.transaction_date
   AND src.currency_code = tgt.currency_code
WHERE src.source_count <> tgt.target_count
   OR src.source_amount <> tgt.target_amount
   OR src.source_count IS NULL
   OR tgt.target_count IS NULL;
```

---

# 13. Minus / Except / Anti-Join Testing

Different databases support different syntax.

| Database | Set Difference Syntax |
|---|---|
| Oracle | `MINUS` |
| PostgreSQL | `EXCEPT` |
| SQL Server | `EXCEPT` |
| Spark SQL | `EXCEPT` / `EXCEPT DISTINCT` |
| Hive | Support depends on version and distribution |

---

## Using EXCEPT

Find rows in source but not target:

```sql
SELECT transaction_id, account_id, customer_id, transaction_date, amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

EXCEPT

SELECT transaction_id, account_id, customer_id, transaction_date, amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

---

## Using MINUS

Oracle-style:

```sql
SELECT transaction_id, account_id, customer_id, transaction_date, amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

MINUS

SELECT transaction_id, account_id, customer_id, transaction_date, amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

---

## Using anti-join

```sql
SELECT s.*
FROM source_transaction s
LEFT JOIN target_transaction_fact t
    ON s.transaction_id = t.transaction_id
WHERE t.transaction_id IS NULL
  AND s.batch_id = 'BATCH_20260525';
```

This is often easier to explain in interviews than `EXCEPT`.

---

## Find target records not present in source

```sql
SELECT t.*
FROM target_transaction_fact t
LEFT JOIN source_transaction s
    ON t.transaction_id = s.transaction_id
WHERE s.transaction_id IS NULL
  AND t.batch_id = 'BATCH_20260525';
```

---

# 14. Window Functions

Window functions are heavily used in data QA, especially for duplicate checks, latest-record logic, and SCD testing.

## ROW_NUMBER

```sql
SELECT
    customer_id,
    email,
    updated_timestamp,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id
        ORDER BY updated_timestamp DESC
    ) AS rn
FROM source_customer;
```

This assigns row numbers inside each customer group.

---

## Get latest record per customer

```sql
SELECT *
FROM (
    SELECT
        customer_id,
        email,
        customer_status,
        updated_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM source_customer
) x
WHERE rn = 1;
```

---

## Find duplicate records using ROW_NUMBER

```sql
SELECT *
FROM (
    SELECT
        customer_id,
        email,
        updated_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM source_customer
) x
WHERE rn > 1;
```

---

## RANK vs DENSE_RANK vs ROW_NUMBER

```sql
SELECT
    customer_id,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num,
    RANK() OVER (ORDER BY amount DESC) AS rank_num,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank_num
FROM source_transaction;
```

Explanation:

| Function | Behavior |
|---|---|
| `ROW_NUMBER()` | Gives a unique sequence even when values tie |
| `RANK()` | Gives same rank for ties and skips the next rank |
| `DENSE_RANK()` | Gives same rank for ties without skipping ranks |

---

## Running total

```sql
SELECT
    account_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY account_id
        ORDER BY transaction_date
    ) AS running_balance_change
FROM source_transaction;
```

This is useful for account-level transaction analysis.

---

# 15. CTEs and Subqueries

CTEs make SQL easier to read.

## Basic CTE

```sql
WITH active_customers AS (
    SELECT *
    FROM source_customer
    WHERE customer_status = 'ACTIVE'
)
SELECT COUNT(*) AS active_customer_count
FROM active_customers;
```

---

## CTE for source-target reconciliation

```sql
WITH source_counts AS (
    SELECT batch_id, COUNT(*) AS source_count
    FROM source_transaction
    GROUP BY batch_id
),
target_counts AS (
    SELECT batch_id, COUNT(*) AS target_count
    FROM target_transaction_fact
    GROUP BY batch_id
)
SELECT
    COALESCE(s.batch_id, t.batch_id) AS batch_id,
    s.source_count,
    t.target_count,
    CASE
        WHEN s.source_count = t.target_count THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM source_counts s
FULL OUTER JOIN target_counts t
    ON s.batch_id = t.batch_id;
```

---

## Subquery example

```sql
SELECT *
FROM source_transaction
WHERE amount > (
    SELECT AVG(amount)
    FROM source_transaction
);
```

---

## EXISTS example

```sql
SELECT s.*
FROM source_customer s
WHERE EXISTS (
    SELECT 1
    FROM target_customer_dim t
    WHERE t.customer_id = s.customer_id
);
```

---

## NOT EXISTS example

```sql
SELECT s.*
FROM source_customer s
WHERE NOT EXISTS (
    SELECT 1
    FROM target_customer_dim t
    WHERE t.customer_id = s.customer_id
);
```

This finds source customers missing from target.

---

# 16. Case Statements and Business Rule Testing

`CASE WHEN` is commonly used for validation.

## Validate transaction amount rule

Rule:

- `PAYMENT`, `DEPOSIT`, and `TRANSFER` should have positive amounts.
- `REVERSAL` may have negative amounts.

```sql
SELECT
    transaction_id,
    transaction_type,
    amount,
    CASE
        WHEN transaction_type IN ('PAYMENT', 'DEPOSIT', 'TRANSFER') AND amount > 0 THEN 'PASS'
        WHEN transaction_type = 'REVERSAL' AND amount < 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS validation_status
FROM source_transaction;
```

---

## Return only failed records

```sql
SELECT *
FROM (
    SELECT
        transaction_id,
        transaction_type,
        amount,
        CASE
            WHEN transaction_type IN ('PAYMENT', 'DEPOSIT', 'TRANSFER') AND amount > 0 THEN 'PASS'
            WHEN transaction_type = 'REVERSAL' AND amount < 0 THEN 'PASS'
            ELSE 'FAIL'
        END AS validation_status
    FROM source_transaction
) x
WHERE validation_status = 'FAIL';
```

---

## Multiple validation checks in one query

```sql
SELECT
    transaction_id,
    CASE WHEN customer_id IS NULL THEN 'FAIL' ELSE 'PASS' END AS customer_id_check,
    CASE WHEN amount IS NULL THEN 'FAIL' ELSE 'PASS' END AS amount_null_check,
    CASE WHEN amount = 0 THEN 'WARNING' ELSE 'PASS' END AS zero_amount_check,
    CASE WHEN currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP') THEN 'FAIL' ELSE 'PASS' END AS currency_check
FROM source_transaction;
```

---

# 17. Incremental Load and CDC Testing

Incremental load means only new or changed records should be processed.

## Validate records loaded after a timestamp

```sql
SELECT COUNT(*) AS incremental_source_count
FROM source_customer
WHERE updated_timestamp > TIMESTAMP '2026-05-24 00:00:00';
```

Target:

```sql
SELECT COUNT(*) AS incremental_target_count
FROM target_customer_dim
WHERE updated_timestamp > TIMESTAMP '2026-05-24 00:00:00';
```

---

## Check if only expected batch loaded

```sql
SELECT DISTINCT batch_id
FROM target_transaction_fact
WHERE CAST(load_timestamp AS DATE) = DATE '2026-05-25';
```

If unexpected batch IDs appear, the pipeline may have loaded the wrong data.

---

## CDC operation validation

Assume a CDC table has operation codes:

- `I` = Insert
- `U` = Update
- `D` = Delete

```sql
SELECT operation_code, COUNT(*) AS record_count
FROM customer_cdc_stage
GROUP BY operation_code;
```

---

## Validate deletes were handled

```sql
SELECT s.customer_id
FROM customer_cdc_stage s
JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.operation_code = 'D'
  AND t.is_current = 'Y';
```

This returns records that were marked as deleted in CDC but are still active in target.

---

# 18. SCD Type 1 and Type 2 Testing

Slowly Changing Dimension testing appears often in data warehouse QA interviews.

---

## SCD Type 1

SCD Type 1 overwrites old values.

Example rule: If customer email changes, target should keep only the latest email.

```sql
SELECT
    s.customer_id,
    s.email AS source_email,
    t.email AS target_email
FROM source_customer s
JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE s.email <> t.email
  AND t.is_current = 'Y';
```

For SCD Type 1, this query should return zero mismatches after load.

---

## SCD Type 2

SCD Type 2 keeps historical records.

Important checks:

- Only one current record per customer.
- Old record should have `is_current = 'N'`.
- Current record should have `is_current = 'Y'`.
- Effective dates should not overlap.
- Current record often has high end date such as `9999-12-31`.

---

## Check only one current record per customer

```sql
SELECT customer_id, COUNT(*) AS current_record_count
FROM target_customer_dim
WHERE is_current = 'Y'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

Expected result: zero rows.

---

## Check current records have open-ended end date

```sql
SELECT *
FROM target_customer_dim
WHERE is_current = 'Y'
  AND effective_end_date <> DATE '9999-12-31';
```

Expected result: zero rows.

---

## Check historical records do not have open-ended end date

```sql
SELECT *
FROM target_customer_dim
WHERE is_current = 'N'
  AND effective_end_date = DATE '9999-12-31';
```

Expected result: zero rows.

---

## Check overlapping effective date ranges

```sql
SELECT
    a.customer_id,
    a.effective_start_date AS a_start,
    a.effective_end_date AS a_end,
    b.effective_start_date AS b_start,
    b.effective_end_date AS b_end
FROM target_customer_dim a
JOIN target_customer_dim b
    ON a.customer_id = b.customer_id
   AND a.customer_key <> b.customer_key
WHERE a.effective_start_date <= b.effective_end_date
  AND b.effective_start_date <= a.effective_end_date;
```

This finds overlapping SCD Type 2 date ranges.

---

# 19. Big Data SQL: Hive and Spark SQL

In Big Data QA, SQL may run on:

- Hive
- Spark SQL
- Databricks SQL
- Impala
- Presto/Trino
- Cloud data warehouses
- Traditional RDBMS systems

The syntax is mostly SQL-like, but functions and performance behavior can vary.

---

## Hive/Spark table count

```sql
SELECT COUNT(*)
FROM database_name.transaction_table
WHERE batch_id = 'BATCH_20260525';
```

---

## Partition validation

Big Data tables are often partitioned by date, batch, or region.

```sql
SHOW PARTITIONS target_transaction_fact;
```

Hive/Spark syntax may vary.

---

## Query a specific partition

```sql
SELECT COUNT(*)
FROM target_transaction_fact
WHERE business_date = DATE '2026-05-25';
```

This is better than scanning the whole table.

---

## Validate partition has data

```sql
SELECT
    business_date,
    COUNT(*) AS record_count
FROM target_transaction_fact
WHERE business_date = DATE '2026-05-25'
GROUP BY business_date;
```

---

## Spark SQL EXPLAIN

```sql
EXPLAIN
SELECT customer_id, COUNT(*)
FROM target_transaction_fact
GROUP BY customer_id;
```

Use `EXPLAIN` to understand query execution plans, especially for large joins.

---

## Hive/Spark RLIKE

```sql
SELECT *
FROM raw_customer_stage
WHERE email NOT RLIKE '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$';
```

Use regex carefully; exact support can vary.

---

## Big Data QA best practices

When writing SQL on large datasets:

- Filter by partition columns.
- Avoid `SELECT *` on huge tables.
- Validate using grouped aggregations first.
- Compare counts before row-level comparison.
- Use sampling carefully but do not rely only on sampling.
- Use joins only on necessary columns.
- Avoid functions on partition columns in `WHERE` if they prevent partition pruning.
- Check rejected records and audit tables.
- Save validation outputs for evidence.

---

# 20. SQL Performance Concepts for QA

You do not need to be a database performance expert, but you should understand the basics.

## Indexes

Indexes help databases find rows faster.

QA interview answer:

> If a validation query is slow on a relational database, I would check whether join keys and filter columns are indexed, whether the query is scanning unnecessary data, and whether I can filter by batch ID or date.

---

## Partitioning

Big Data tables are often partitioned.

Example:

```sql
WHERE business_date = DATE '2026-05-25'
```

This can reduce the amount of data scanned.

---

## Avoid functions on partition columns

Less efficient:

```sql
WHERE CAST(load_timestamp AS DATE) = DATE '2026-05-25'
```

More efficient if a partition column exists:

```sql
WHERE business_date = DATE '2026-05-25'
```

---

## Filter early

Instead of joining all data:

```sql
SELECT *
FROM source_transaction s
JOIN target_transaction_fact t
    ON s.transaction_id = t.transaction_id
WHERE s.batch_id = 'BATCH_20260525';
```

Prefer filtering first:

```sql
WITH src AS (
    SELECT *
    FROM source_transaction
    WHERE batch_id = 'BATCH_20260525'
),
tgt AS (
    SELECT *
    FROM target_transaction_fact
    WHERE batch_id = 'BATCH_20260525'
)
SELECT *
FROM src s
JOIN tgt t
    ON s.transaction_id = t.transaction_id;
```

---

## Use counts before full comparison

Start with:

```sql
SELECT COUNT(*) FROM source_transaction WHERE batch_id = 'BATCH_20260525';
SELECT COUNT(*) FROM target_transaction_fact WHERE batch_id = 'BATCH_20260525';
```

Then go deeper only if needed.

---

# 21. Banking and Financial Data Validation Scenarios

## Scenario 1: Transaction amount reconciliation

Question:

> How would you validate transaction amounts between source and target?

Answer:

```sql
SELECT
    'source' AS dataset_name,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

UNION ALL

SELECT
    'target' AS dataset_name,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

Explain:

> I would compare both count and amount totals. If they do not match, I would group by date, currency, transaction type, or account to isolate the mismatch.

---

## Scenario 2: Invalid negative amounts

```sql
SELECT *
FROM source_transaction
WHERE amount < 0
  AND transaction_type <> 'REVERSAL';
```

---

## Scenario 3: Missing account IDs

```sql
SELECT COUNT(*) AS missing_account_count
FROM source_transaction
WHERE account_id IS NULL
   OR TRIM(account_id) = '';
```

---

## Scenario 4: Currency code validation

```sql
SELECT currency_code, COUNT(*) AS invalid_count
FROM source_transaction
WHERE currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP')
GROUP BY currency_code;
```

---

## Scenario 5: Trade date cannot be greater than processing date

```sql
SELECT *
FROM trade_stage
WHERE trade_date > processing_date;
```

---

## Scenario 6: Customer should have only one active profile

```sql
SELECT customer_id, COUNT(*) AS active_profile_count
FROM customer_profile
WHERE status = 'ACTIVE'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Scenario 7: Audit table validation

Assume an ETL audit table:

```sql
etl_audit
---------
job_name
batch_id
source_count
target_count
reject_count
job_status
start_time
end_time
```

Query:

```sql
SELECT *
FROM etl_audit
WHERE batch_id = 'BATCH_20260525'
  AND job_status <> 'SUCCESS';
```

Check reconciliation:

```sql
SELECT *
FROM etl_audit
WHERE batch_id = 'BATCH_20260525'
  AND source_count <> target_count + reject_count;
```

---

# 22. SQL Interview Questions and Model Answers

## Q1. What is the difference between WHERE and HAVING?

**Answer:**  
`WHERE` filters rows before aggregation. `HAVING` filters grouped results after aggregation.

Example:

```sql
SELECT customer_id, COUNT(*) AS record_count
FROM source_customer
WHERE customer_status = 'ACTIVE'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Q2. How do you find duplicate records?

```sql
SELECT customer_id, COUNT(*) AS duplicate_count
FROM source_customer
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Q3. How do you find records in source but missing in target?

```sql
SELECT s.*
FROM source_customer s
LEFT JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE t.customer_id IS NULL;
```

---

## Q4. Difference between INNER JOIN and LEFT JOIN?

**Answer:**  
`INNER JOIN` returns only matching records from both tables.  
`LEFT JOIN` returns all records from the left table and matching records from the right table. If there is no match, right-side columns are null.

---

## Q5. How do you validate source-to-target count?

```sql
SELECT 'source' AS table_name, COUNT(*) AS record_count
FROM source_transaction
WHERE batch_id = 'BATCH_20260525'

UNION ALL

SELECT 'target' AS table_name, COUNT(*) AS record_count
FROM target_transaction_fact
WHERE batch_id = 'BATCH_20260525';
```

---

## Q6. How do you validate source-to-target data values?

```sql
SELECT
    s.transaction_id,
    s.amount AS source_amount,
    t.amount AS target_amount
FROM source_transaction s
JOIN target_transaction_fact t
    ON s.transaction_id = t.transaction_id
WHERE s.amount <> t.amount;
```

For null-safe comparison, use database-specific functions or explicit null handling:

```sql
WHERE COALESCE(s.amount, 0) <> COALESCE(t.amount, 0);
```

Be careful using `0` for null handling if `0` is a valid business value.

---

## Q7. What is ROW_NUMBER used for?

**Answer:**  
`ROW_NUMBER()` assigns a unique sequence number within a partition. It is useful for finding duplicates, selecting the latest record, and testing SCD logic.

```sql
SELECT *
FROM (
    SELECT
        customer_id,
        updated_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM source_customer
) x
WHERE rn = 1;
```

---

## Q8. What is the difference between DELETE, TRUNCATE, and DROP?

| Command | Meaning |
|---|---|
| `DELETE` | Removes selected rows; can use `WHERE` |
| `TRUNCATE` | Removes all rows quickly |
| `DROP` | Removes the table structure itself |

In a QA interview, mention that permissions and transaction behavior vary by database.

---

## Q9. What is a primary key?

A primary key uniquely identifies each row in a table. In QA, primary keys are validated for:

- Nulls
- Duplicates
- Missing values between source and target
- Incorrect mappings

---

## Q10. What is a foreign key?

A foreign key links one table to another. In QA, it is used for referential integrity validation.

Example:

```sql
SELECT t.*
FROM source_transaction t
LEFT JOIN source_customer c
    ON t.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

This finds transactions with customer IDs that do not exist in the customer table.

---

## Q11. How would you test a data pipeline?

Strong answer:

> I would start with file or table availability checks, then verify counts between source and target. Next, I would validate schema, nulls, duplicates, data types, and business rules. After that, I would perform source-to-target field-level comparison, aggregate reconciliation, reject-table validation, and audit-table validation. For large datasets, I would use partition filters, grouped comparisons, and targeted mismatch queries.

---

## Q12. How do you handle a query that is too slow?

Strong answer:

> I would first check whether the query is scanning too much data. I would filter by batch ID or business date, select only required columns, check joins, use aggregations before row-level comparison, and review the execution plan if available. In Hive or Spark SQL, I would make sure partition filters are used.

---

# 23. Hands-On SQL Practice Exercises

## Exercise 1: Find duplicate customer IDs

Write a query to find duplicate customer IDs in `source_customer`.

Expected solution:

```sql
SELECT customer_id, COUNT(*) AS duplicate_count
FROM source_customer
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Exercise 2: Find source records missing in target

```sql
SELECT s.customer_id
FROM source_customer s
LEFT JOIN target_customer_dim t
    ON s.customer_id = t.customer_id
WHERE t.customer_id IS NULL;
```

---

## Exercise 3: Validate count and amount by currency

```sql
SELECT
    currency_code,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM source_transaction
GROUP BY currency_code;
```

---

## Exercise 4: Find invalid transaction records

Rules:

- `transaction_id` cannot be null.
- `account_id` cannot be null.
- `amount` cannot be null.
- `currency_code` must be one of `CAD`, `USD`, `EUR`, `GBP`.

```sql
SELECT *
FROM source_transaction
WHERE transaction_id IS NULL
   OR account_id IS NULL
   OR amount IS NULL
   OR currency_code NOT IN ('CAD', 'USD', 'EUR', 'GBP');
```

---

## Exercise 5: Validate source and target amounts by transaction type

```sql
WITH src AS (
    SELECT transaction_type, SUM(amount) AS source_amount
    FROM source_transaction
    WHERE batch_id = 'BATCH_20260525'
    GROUP BY transaction_type
),
tgt AS (
    SELECT transaction_type, SUM(amount) AS target_amount
    FROM target_transaction_fact
    WHERE batch_id = 'BATCH_20260525'
    GROUP BY transaction_type
)
SELECT
    COALESCE(src.transaction_type, tgt.transaction_type) AS transaction_type,
    src.source_amount,
    tgt.target_amount,
    CASE
        WHEN src.source_amount = tgt.target_amount THEN 'PASS'
        ELSE 'FAIL'
    END AS validation_status
FROM src
FULL OUTER JOIN tgt
    ON src.transaction_type = tgt.transaction_type;
```

---

## Exercise 6: Find latest customer record

```sql
SELECT *
FROM (
    SELECT
        customer_id,
        email,
        updated_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM source_customer
) x
WHERE rn = 1;
```

---

## Exercise 7: Find multiple current SCD records

```sql
SELECT customer_id, COUNT(*) AS current_record_count
FROM target_customer_dim
WHERE is_current = 'Y'
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Exercise 8: Validate reject count

```sql
SELECT
    a.batch_id,
    a.source_count,
    a.target_count,
    a.reject_count,
    CASE
        WHEN a.source_count = a.target_count + a.reject_count THEN 'PASS'
        ELSE 'FAIL'
    END AS validation_status
FROM etl_audit a
WHERE a.batch_id = 'BATCH_20260525';
```

---

# 24. 7-Day SQL Preparation Plan

## Day 1: SQL basics

Practice:

- `SELECT`
- `WHERE`
- `ORDER BY`
- `DISTINCT`
- `COUNT`
- `LIMIT` / `FETCH FIRST`

Goal: Be comfortable reading and filtering data.

---

## Day 2: Aggregations

Practice:

- `GROUP BY`
- `HAVING`
- `COUNT`
- `SUM`
- `MIN`
- `MAX`
- `AVG`

Goal: Be able to do count and sum reconciliation.

---

## Day 3: Joins

Practice:

- `INNER JOIN`
- `LEFT JOIN`
- `FULL OUTER JOIN`
- Missing-record checks
- Source-to-target comparisons

Goal: Be able to find missing and mismatched records.

---

## Day 4: Data quality checks

Practice:

- Null checks
- Blank checks
- Duplicate checks
- Invalid value checks
- Date checks
- Format checks

Goal: Be able to write QA validation queries quickly.

---

## Day 5: Window functions and CTEs

Practice:

- `ROW_NUMBER`
- `RANK`
- `DENSE_RANK`
- Latest-record selection
- Duplicate ranking
- CTE-based validation queries

Goal: Be comfortable with intermediate SQL.

---

## Day 6: Big Data SQL

Practice:

- Hive/Spark SQL basics
- Partition filters
- `EXPLAIN`
- Large-table validation approach
- Batch-level reconciliation

Goal: Learn how SQL changes when datasets are very large.

---

## Day 7: Mock interview

Practice explaining:

- Source-to-target testing
- Data quality validation
- Count and amount reconciliation
- Duplicate handling
- Null handling
- SCD testing
- Incremental load testing
- How to debug failed ETL loads

Goal: Give structured answers, not just queries.

---

# 25. Final Cheat Sheet

## Most important SQL patterns

### Count check

```sql
SELECT COUNT(*) FROM table_name;
```

### Duplicate check

```sql
SELECT key_column, COUNT(*)
FROM table_name
GROUP BY key_column
HAVING COUNT(*) > 1;
```

### Null check

```sql
SELECT *
FROM table_name
WHERE key_column IS NULL;
```

### Missing in target

```sql
SELECT s.*
FROM source_table s
LEFT JOIN target_table t
    ON s.key_column = t.key_column
WHERE t.key_column IS NULL;
```

### Mismatched values

```sql
SELECT
    s.key_column,
    s.amount AS source_amount,
    t.amount AS target_amount
FROM source_table s
JOIN target_table t
    ON s.key_column = t.key_column
WHERE s.amount <> t.amount;
```

### Group reconciliation

```sql
SELECT
    business_date,
    currency_code,
    COUNT(*) AS record_count,
    SUM(amount) AS total_amount
FROM transaction_table
GROUP BY business_date, currency_code;
```

### Latest record

```sql
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY business_key
            ORDER BY updated_timestamp DESC
        ) AS rn
    FROM table_name
) x
WHERE rn = 1;
```

### SCD current record check

```sql
SELECT business_key, COUNT(*)
FROM dimension_table
WHERE is_current = 'Y'
GROUP BY business_key
HAVING COUNT(*) > 1;
```

---

## Interview words to use

Use these phrases in your answers:

- Source-to-target validation
- Count reconciliation
- Amount reconciliation
- Field-level comparison
- Duplicate validation
- Null validation
- Referential integrity
- Business rule validation
- Reject-table validation
- Audit-table validation
- Batch-level validation
- Partition-level validation
- Incremental load testing
- SCD Type 1 and Type 2 validation
- Root cause analysis
- Data quality controls
- Production-safe validation

---

## Strong closing answer

If asked, **“How do you use SQL in Big Data QA?”**, you can say:

> I use SQL to validate data across the pipeline from source to target. I start with count checks, file/table availability, and batch-level validation. Then I validate nulls, duplicates, data types, domain values, and business rules. For source-to-target testing, I use joins, anti-joins, except/minus queries, aggregate reconciliation, and field-level comparisons. For Big Data platforms like Hive or Spark SQL, I make sure to filter by partitions and validate at both aggregate and detailed levels. In a banking environment, I also focus on auditability, reconciliation, and avoiding exposure of sensitive customer or transaction data.

---

# 26. Reference Links

These are useful for deeper study:

- RBC Careers: Quality Engineer and Data Engineer postings frequently mention SQL, Unix/Linux, Python, Big Data/Hadoop, Spark, large-scale databases, automated testing, and data pipeline skills.
- Apache Spark SQL documentation: https://spark.apache.org/docs/latest/sql-ref-syntax.html
- Apache Hive language manual: https://hive.apache.org/docs/latest/language/
- PostgreSQL documentation: https://www.postgresql.org/docs/
- Microsoft SQL Server documentation: https://learn.microsoft.com/en-us/sql/
- Oracle SQL documentation: https://docs.oracle.com/en/database/oracle/oracle-database/
