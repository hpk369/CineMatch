# Unix/Shell Scripting Interview Guide for Big Data QA Roles at RBC

**Target role:** Big Data QA / Quality Engineer / Data Quality Engineer  
**Primary focus:** Unix/Linux, shell scripting, log analysis, batch validation, HDFS/Hadoop, SQL-adjacent QA checks, and production support thinking  
**Audience:** Candidate preparing for RBC-style QA interviews in banking/financial data environments  
**Status:** Interview-prep guide, not an official RBC document

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)
2. [What RBC-Style Big Data QA Interviews Usually Test](#2-what-rbc-style-big-data-qa-interviews-usually-test)
3. [Unix/Linux Fundamentals](#3-unixlinux-fundamentals)
4. [File Inspection and Large File Handling](#4-file-inspection-and-large-file-handling)
5. [Text Search With grep](#5-text-search-with-grep)
6. [Sorting, De-duplication, and Counting](#6-sorting-de-duplication-and-counting)
7. [Column Processing With cut and awk](#7-column-processing-with-cut-and-awk)
8. [Text Transformation With sed](#8-text-transformation-with-sed)
9. [Shell Scripting Fundamentals](#9-shell-scripting-fundamentals)
10. [Production-Ready Shell Script Practices](#10-production-ready-shell-script-practices)
11. [Big Data QA Validation Scenarios](#11-big-data-qa-validation-scenarios)
12. [HDFS/Hadoop Commands for QA](#12-hdfshadoop-commands-for-qa)
13. [Log Analysis for Batch Jobs](#13-log-analysis-for-batch-jobs)
14. [Data Reconciliation Patterns](#14-data-reconciliation-patterns)
15. [Banking/Data Quality Checks](#15-bankingdata-quality-checks)
16. [Mock Interview Questions and Strong Answers](#16-mock-interview-questions-and-strong-answers)
17. [Hands-On Coding Exercises](#17-hands-on-coding-exercises)
18. [Mini Project: QA Validation Framework](#18-mini-project-qa-validation-framework)
19. [7-Day Preparation Plan](#19-7-day-preparation-plan)
20. [Final Cheat Sheet](#20-final-cheat-sheet)
21. [References](#21-references)

---

# 1. How to Use This Guide

This guide is designed to prepare you for Unix and shell scripting questions in a **Big Data QA** or **Quality Engineer** interview at a bank such as RBC.

You should not only memorize commands. The goal is to explain how you would use Unix scripting to validate real data pipelines.

A good interview response usually has three layers:

1. **Command knowledge**  
   Example: `grep`, `awk`, `sort`, `uniq`, `wc`, `find`, `hdfs dfs`.

2. **QA thinking**  
   Example: file arrival, record count, duplicate check, null validation, source-to-target reconciliation.

3. **Production mindset**  
   Example: logging, exit codes, auditability, security, avoiding sensitive data exposure, and handling failures gracefully.

---

# 2. What RBC-Style Big Data QA Interviews Usually Test

Public RBC Quality Engineer postings commonly emphasize skills around **Unix/Linux**, **Big Data/Hadoop**, **SQL**, **automated testing**, production troubleshooting, and quality engineering in data environments.

For a Big Data QA role, you may be tested on:

- Unix/Linux commands
- Shell scripting basics
- Batch job validation
- File validation
- HDFS/Hadoop file operations
- Log analysis
- SQL/data reconciliation concepts
- Automation mindset
- Production support thinking
- Communication of failures and root cause

## What the interviewer wants to know

They may ask:

- Can you work on a Unix/Linux server confidently?
- Can you debug a failed batch job?
- Can you check whether a file arrived correctly?
- Can you validate source and target counts?
- Can you write a shell script with parameters?
- Can you scan logs for errors?
- Can you work with large files without opening them in an editor?
- Can you use HDFS commands in a Hadoop-based environment?
- Can you make your validation repeatable and auditable?

## Strong positioning statement

Use this kind of answer in interviews:

> I use Unix and shell scripting to automate repeatable QA checks such as file arrival validation, record count validation, duplicate detection, null checks, column count checks, schema checks, log scanning, and source-to-target reconciliation. For Big Data pipelines, I combine Unix commands, HDFS commands, SQL queries, and shell scripts to produce clear pass/fail validation reports.

---

# 3. Unix/Linux Fundamentals

## 3.1 Basic navigation commands

| Command | Purpose |
|---|---|
| `pwd` | Show current directory |
| `ls` | List files |
| `ls -ltr` | List files by modification time, oldest first |
| `cd /path` | Change directory |
| `mkdir folder` | Create directory |
| `touch file.txt` | Create empty file or update timestamp |
| `cp source target` | Copy file |
| `mv old new` | Move or rename file |
| `rm file` | Remove file |
| `rm -r folder` | Remove directory recursively |

## 3.2 Common interview questions

### Q: What does `ls -ltr` do?

`ls -ltr` lists files in long format and sorts them by modification time in reverse order, so older files appear first and newer files appear near the bottom.

This is useful in QA/support because you often need to check the most recently generated file.

```bash
ls -ltr /data/inbound
```

### Q: How do you check your current directory?

```bash
pwd
```

### Q: How do you create a directory for QA outputs?

```bash
mkdir -p /data/qa/reports
```

The `-p` option creates parent directories if they do not already exist.

### Q: How do you check hidden files?

```bash
ls -la
```

---

# 4. File Inspection and Large File Handling

Big Data QA often involves files that are too large to open manually. You need to inspect them efficiently.

## 4.1 Basic file inspection commands

| Command | Purpose |
|---|---|
| `cat file` | Print whole file; use only for small files |
| `less file` | View file page by page |
| `head file` | Show first 10 lines |
| `tail file` | Show last 10 lines |
| `tail -f file` | Follow live updates to a file |
| `wc -l file` | Count lines |
| `file filename` | Identify file type |

## 4.2 Examples

### Check first few rows

```bash
head customer_data.csv
```

### Check first 5 rows

```bash
head -5 customer_data.csv
```

### Check last 20 rows

```bash
tail -20 customer_data.csv
```

### Monitor a running log

```bash
tail -f batch_job.log
```

### Count total lines

```bash
wc -l customer_data.csv
```

### Count records excluding header

```bash
tail -n +2 customer_data.csv | wc -l
```

`tail -n +2` starts reading from line 2, so it excludes the header.

## 4.3 Interview explanation

If asked how you validate a large file:

> I would avoid opening the file manually. I would use `head` to check the header and sample data, `tail` to check trailer or final records, `wc -l` for record count, `awk` for column-level checks, and `grep` for patterns or invalid values.

---

# 5. Text Search With grep

`grep` is one of the most important commands for QA and production support.

## 5.1 Common grep options

| Command | Purpose |
|---|---|
| `grep "text" file` | Search for exact text |
| `grep -i "text" file` | Case-insensitive search |
| `grep -v "text" file` | Exclude matching lines |
| `grep -n "text" file` | Show line numbers |
| `grep -c "text" file` | Count matching lines |
| `grep -E "A|B" file` | Extended regex; match A or B |
| `grep -r "text" folder` | Recursive search |

## 5.2 Log analysis examples

### Find errors

```bash
grep -i "error" application.log
```

### Find multiple failure keywords

```bash
grep -Ei "error|exception|failed|fatal|aborted" application.log
```

### Count errors

```bash
grep -Ei "error|exception|failed|fatal" application.log | wc -l
```

### Show line numbers for failures

```bash
grep -nEi "error|exception|failed|fatal" application.log
```

### Exclude informational messages

```bash
grep -v "INFO" application.log
```

### Search in all logs under a directory

```bash
grep -rEi "error|exception|failed|fatal" /data/logs
```

## 5.3 Strong interview answer

> To analyze a failed batch job, I would first check the job log using `grep -Ei "error|exception|failed|fatal"`. Then I would check timestamps around the error using `sed` or `tail`, identify whether the issue is data-related, permission-related, file-missing, schema-related, or environment-related, and then validate the input and output files.

---

# 6. Sorting, De-duplication, and Counting

These are core commands for duplicate checks and frequency analysis.

## 6.1 sort

```bash
sort file.txt
```

Sort by first column numerically:

```bash
sort -n file.txt
```

Sort in reverse order:

```bash
sort -r file.txt
```

Sort CSV by second column:

```bash
sort -t',' -k2,2 customer.csv
```

## 6.2 uniq

`uniq` works best after sorting.

### Remove duplicate lines

```bash
sort file.txt | uniq
```

### Find duplicate lines

```bash
sort file.txt | uniq -d
```

### Count occurrences

```bash
sort file.txt | uniq -c
```

### Sort counts from highest to lowest

```bash
sort file.txt | uniq -c | sort -nr
```

## 6.3 QA examples

### Find duplicate customer IDs from first column

```bash
awk -F',' 'NR > 1 {print $1}' customer.csv | sort | uniq -d
```

### Count customer ID frequency

```bash
awk -F',' 'NR > 1 {print $1}' customer.csv | sort | uniq -c | sort -nr
```

### Find duplicate full records

```bash
sort customer.csv | uniq -d
```

## 6.4 Interview explanation

> To find duplicate IDs, I would extract the key column using `awk` or `cut`, sort it, and then use `uniq -d` to display duplicates. If I need counts, I would use `uniq -c`.

---

# 7. Column Processing With cut and awk

Big Data QA often requires validating specific columns. For CSV or pipe-delimited files, `cut` and `awk` are very useful.

---

## 7.1 cut

### Extract first column from comma-delimited file

```bash
cut -d',' -f1 customer.csv
```

### Extract first and third columns

```bash
cut -d',' -f1,3 customer.csv
```

### Extract fields from pipe-delimited file

```bash
cut -d'|' -f1,2 transactions.txt
```

## 7.2 awk basics

`awk` processes files line by line. It is especially useful for field-based validation.

Basic syntax:

```bash
awk 'pattern { action }' file
```

For CSV files:

```bash
awk -F',' '{print $1}' customer.csv
```

Here:

- `-F','` means comma is the field separator.
- `$1` means first column.
- `$2` means second column.
- `NR` means current record number.
- `NF` means number of fields in the current record.

## 7.3 Common awk commands for QA

### Print first column

```bash
awk -F',' '{print $1}' customer.csv
```

### Print first and second columns

```bash
awk -F',' '{print $1, $2}' customer.csv
```

### Skip header

```bash
awk -F',' 'NR > 1 {print $0}' customer.csv
```

### Print line number and row

```bash
awk -F',' '{print NR, $0}' customer.csv
```

### Check invalid column count

```bash
awk -F',' 'NF != 5 {print NR, $0}' customer.csv
```

### Count bad rows with invalid column count

```bash
awk -F',' 'NF != 5 {count++} END {print count+0}' customer.csv
```

### Find blank customer IDs in column 1

```bash
awk -F',' 'NR > 1 && $1 == "" {print NR, $0}' customer.csv
```

### Count blank values in column 3

```bash
awk -F',' 'NR > 1 && $3 == "" {count++} END {print count+0}' customer.csv
```

### Validate amount greater than 10000

```bash
awk -F',' 'NR > 1 && $4 > 10000 {print NR, $0}' transactions.csv
```

### Calculate sum of amount column

```bash
awk -F',' 'NR > 1 {sum += $4} END {print sum}' transactions.csv
```

### Find minimum and maximum transaction amount

```bash
awk -F',' 'NR==2 {min=$4; max=$4} NR>1 {if($4<min) min=$4; if($4>max) max=$4} END {print "MIN=" min, "MAX=" max}' transactions.csv
```

### Validate date format YYYY-MM-DD in column 5

```bash
awk -F',' 'NR > 1 && $5 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ {print NR, $0}' transactions.csv
```

### Validate numeric amount in column 4

```bash
awk -F',' 'NR > 1 && $4 !~ /^-?[0-9]+(\.[0-9]+)?$/ {print NR, $0}' transactions.csv
```

## 7.4 Interview explanation

> I use `awk` for column-level validation. For example, I can validate field counts using `NF`, skip headers using `NR > 1`, check nulls, validate date formats with regex, calculate sums, and identify records that break business rules.

---

# 8. Text Transformation With sed

`sed` is useful for simple replacements, extracting line ranges, and cleaning text.

## 8.1 Common sed examples

### Replace old value with new value

```bash
sed 's/old/new/g' file.txt
```

### Convert pipe-delimited file to comma-delimited

```bash
sed 's/|/,/g' input.txt > output.csv
```

### Print lines 100 to 120

```bash
sed -n '100,120p' application.log
```

### Remove blank lines

```bash
sed '/^$/d' file.txt
```

### Remove carriage returns from Windows files

```bash
sed 's/\r$//' input.txt > clean_input.txt
```

## 8.2 Interview explanation

> I usually use `sed` when I need quick text substitution or when I need to extract a specific line range from logs. For complex column-based validation, I prefer `awk`.

---

# 9. Shell Scripting Fundamentals

## 9.1 Basic script structure

```bash
#!/bin/bash

echo "Starting QA validation..."

file_name="customer_data.csv"

if [ -f "$file_name" ]; then
    echo "File exists: $file_name"
else
    echo "File does not exist: $file_name"
    exit 1
fi
```

## 9.2 Script arguments

| Variable | Meaning |
|---|---|
| `$0` | Script name |
| `$1` | First argument |
| `$2` | Second argument |
| `$#` | Number of arguments |
| `$?` | Exit code of previous command |
| `$$` | Process ID of current shell |

## 9.3 Example: script with input argument

```bash
#!/bin/bash

input_file=$1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "ERROR: File not found: $input_file"
    exit 1
fi

echo "Record count:"
wc -l "$input_file"
```

Run:

```bash
bash validate_file.sh customer.csv
```

## 9.4 If condition

```bash
if [ -f data.csv ]; then
    echo "File exists"
else
    echo "File missing"
fi
```

## 9.5 For loop

```bash
for file in *.csv
do
    echo "Processing $file"
    wc -l "$file"
done
```

## 9.6 While loop

```bash
while read line
do
    echo "$line"
done < input.txt
```

Better version for preserving spacing and backslashes:

```bash
while IFS= read -r line
do
    echo "$line"
done < input.txt
```

## 9.7 Case statement

```bash
env=$1

case "$env" in
    dev)
        echo "Running in DEV"
        ;;
    qa)
        echo "Running in QA"
        ;;
    prod)
        echo "Running in PROD"
        ;;
    *)
        echo "Invalid environment"
        exit 1
        ;;
esac
```

---

# 10. Production-Ready Shell Script Practices

A strong interview answer is not just about writing a script that works once. It should be safe, readable, and maintainable.

## 10.1 Use strict mode

```bash
set -euo pipefail
```

Meaning:

- `set -e`: exit when a command fails.
- `set -u`: fail when using an undefined variable.
- `set -o pipefail`: fail if any command in a pipeline fails.

## 10.2 Quote variables

Bad:

```bash
rm $file
```

Better:

```bash
rm "$file"
```

Quoting helps prevent issues with spaces, empty variables, and unexpected file names.

## 10.3 Use functions

```bash
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

fail() {
    log "ERROR: $1"
    exit 1
}
```

## 10.4 Write logs

```bash
log_file="qa_validation.log"

echo "Validation started" | tee -a "$log_file"
```

## 10.5 Use meaningful exit codes

```bash
exit 0  # success
exit 1  # failure
```

Schedulers, CI/CD pipelines, and monitoring tools depend on exit codes.

## 10.6 Production-ready validation script template

```bash
#!/bin/bash
set -euo pipefail

log_file="qa_validation.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$log_file"
}

fail() {
    log "FAIL: $1"
    exit 1
}

input_file=${1:-}
expected_columns=${2:-}

if [ -z "$input_file" ] || [ -z "$expected_columns" ]; then
    fail "Usage: $0 <input_file> <expected_columns>"
fi

if [ ! -f "$input_file" ]; then
    fail "File not found: $input_file"
fi

if [ ! -s "$input_file" ]; then
    fail "File is empty: $input_file"
fi

bad_rows=$(awk -F',' -v cols="$expected_columns" 'NF != cols {count++} END {print count+0}' "$input_file")

if [ "$bad_rows" -ne 0 ]; then
    fail "$bad_rows rows have invalid column count"
fi

record_count=$(tail -n +2 "$input_file" | wc -l)

log "PASS: File validation completed"
log "Record count excluding header: $record_count"
exit 0
```

---

# 11. Big Data QA Validation Scenarios

This section is especially important for interviews.

---

## Scenario 1: File exists and is not empty

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}

if [ -z "$file" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

if [ ! -f "$file" ]; then
    echo "FAIL: File does not exist: $file"
    exit 1
fi

if [ ! -s "$file" ]; then
    echo "FAIL: File is empty: $file"
    exit 1
fi

echo "PASS: File exists and is not empty"
```

---

## Scenario 2: Validate expected record count

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}
expected_count=${2:-}

if [ -z "$file" ] || [ -z "$expected_count" ]; then
    echo "Usage: $0 <file> <expected_count>"
    exit 1
fi

actual_count=$(tail -n +2 "$file" | wc -l)

if [ "$actual_count" -eq "$expected_count" ]; then
    echo "PASS: Record count matched. Count=$actual_count"
else
    echo "FAIL: Expected $expected_count but got $actual_count"
    exit 1
fi
```

---

## Scenario 3: Validate column count

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}
expected_columns=${2:-}

if [ -z "$file" ] || [ -z "$expected_columns" ]; then
    echo "Usage: $0 <file> <expected_columns>"
    exit 1
fi

bad_rows=$(awk -F',' -v cols="$expected_columns" 'NR > 1 && NF != cols {count++} END {print count+0}' "$file")

if [ "$bad_rows" -eq 0 ]; then
    echo "PASS: All data rows have $expected_columns columns"
else
    echo "FAIL: $bad_rows data rows have invalid column count"
    awk -F',' -v cols="$expected_columns" 'NR > 1 && NF != cols {print NR, $0}' "$file" | head -20
    exit 1
fi
```

---

## Scenario 4: Find duplicate customer IDs

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}

if [ -z "$file" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

echo "Duplicate customer IDs:"
awk -F',' 'NR > 1 {print $1}' "$file" | sort | uniq -d
```

---

## Scenario 5: Count duplicate keys

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}

if [ -z "$file" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

duplicate_count=$(awk -F',' 'NR > 1 {print $1}' "$file" | sort | uniq -d | wc -l)

echo "Duplicate key count: $duplicate_count"

if [ "$duplicate_count" -gt 0 ]; then
    echo "FAIL: Duplicate keys found"
    exit 1
else
    echo "PASS: No duplicate keys found"
fi
```

---

## Scenario 6: Source and target count comparison

```bash
#!/bin/bash
set -euo pipefail

source_file=${1:-}
target_file=${2:-}

if [ -z "$source_file" ] || [ -z "$target_file" ]; then
    echo "Usage: $0 <source_file> <target_file>"
    exit 1
fi

source_count=$(tail -n +2 "$source_file" | wc -l)
target_count=$(tail -n +2 "$target_file" | wc -l)

echo "Source count: $source_count"
echo "Target count: $target_count"

if [ "$source_count" -eq "$target_count" ]; then
    echo "PASS: Source and target counts match"
else
    echo "FAIL: Source and target counts do not match"
    exit 1
fi
```

---

## Scenario 7: Null check on mandatory fields

Assume file structure:

```text
customer_id,name,email,account_status
```

Script:

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}

if [ -z "$file" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

bad_rows=$(awk -F',' 'NR > 1 && ($1 == "" || $4 == "") {count++} END {print count+0}' "$file")

if [ "$bad_rows" -eq 0 ]; then
    echo "PASS: Mandatory field validation passed"
else
    echo "FAIL: $bad_rows rows have missing customer_id or account_status"
    awk -F',' 'NR > 1 && ($1 == "" || $4 == "") {print NR, $0}' "$file" | head -20
    exit 1
fi
```

---

## Scenario 8: Sum reconciliation

Assume transaction amount is column 4.

```bash
#!/bin/bash
set -euo pipefail

source_file=${1:-}
target_file=${2:-}

source_sum=$(awk -F',' 'NR > 1 {sum += $4} END {printf "%.2f", sum}' "$source_file")
target_sum=$(awk -F',' 'NR > 1 {sum += $4} END {printf "%.2f", sum}' "$target_file")

echo "Source amount sum: $source_sum"
echo "Target amount sum: $target_sum"

if [ "$source_sum" = "$target_sum" ]; then
    echo "PASS: Amount totals match"
else
    echo "FAIL: Amount totals do not match"
    exit 1
fi
```

---

## Scenario 9: Validate date format

Assume transaction date is column 5 and expected format is `YYYY-MM-DD`.

```bash
awk -F',' 'NR > 1 && $5 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ {print NR, $0}' transactions.csv
```

Script version:

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}

invalid_date_count=$(awk -F',' 'NR > 1 && $5 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ {count++} END {print count+0}' "$file")

if [ "$invalid_date_count" -eq 0 ]; then
    echo "PASS: Date format validation passed"
else
    echo "FAIL: $invalid_date_count rows have invalid date format"
    exit 1
fi
```

---

## Scenario 10: Create a QA summary report

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}
report="qa_summary_report_$(date '+%Y%m%d_%H%M%S').txt"

if [ -z "$file" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

total_count=$(tail -n +2 "$file" | wc -l)
duplicate_ids=$(awk -F',' 'NR > 1 {print $1}' "$file" | sort | uniq -d | wc -l)
blank_customer_ids=$(awk -F',' 'NR > 1 && $1 == "" {count++} END {print count+0}' "$file")
invalid_column_count=$(awk -F',' 'NR > 1 && NF != 4 {count++} END {print count+0}' "$file")

{
    echo "QA Summary Report"
    echo "================="
    echo "File Name: $file"
    echo "Generated At: $(date)"
    echo "Total Records Excluding Header: $total_count"
    echo "Duplicate Customer IDs: $duplicate_ids"
    echo "Blank Customer IDs: $blank_customer_ids"
    echo "Invalid Column Count Rows: $invalid_column_count"
} > "$report"

cat "$report"
```

---

# 12. HDFS/Hadoop Commands for QA

For Big Data QA, you may need to validate files in HDFS rather than local Unix directories.

## 12.1 Common HDFS commands

| Command | Purpose |
|---|---|
| `hdfs dfs -ls /path` | List HDFS files |
| `hdfs dfs -cat /path/file` | Print HDFS file content |
| `hdfs dfs -head /path/file` | Show beginning of HDFS file |
| `hdfs dfs -tail /path/file` | Show end of HDFS file |
| `hdfs dfs -du -h /path` | Show HDFS disk usage |
| `hdfs dfs -count /path` | Count directories, files, and bytes |
| `hdfs dfs -mkdir /path` | Create HDFS directory |
| `hdfs dfs -put local /hdfs/path` | Upload local file to HDFS |
| `hdfs dfs -get /hdfs/path local` | Download HDFS file to local |
| `hdfs dfs -rm /path/file` | Remove HDFS file |
| `hdfs dfs -test -e /path` | Test whether path exists |

## 12.2 Check if file exists in HDFS

```bash
hdfs dfs -test -e /data/input/customer.csv

if [ $? -eq 0 ]; then
    echo "File exists in HDFS"
else
    echo "File missing in HDFS"
fi
```

Better script style:

```bash
if hdfs dfs -test -e /data/input/customer.csv; then
    echo "File exists in HDFS"
else
    echo "File missing in HDFS"
    exit 1
fi
```

## 12.3 Count records in HDFS file

```bash
hdfs dfs -cat /data/input/customer.csv | wc -l
```

Excluding header:

```bash
hdfs dfs -cat /data/input/customer.csv | tail -n +2 | wc -l
```

## 12.4 View first rows from HDFS file

```bash
hdfs dfs -cat /data/input/customer.csv | head
```

## 12.5 View last rows from HDFS file

```bash
hdfs dfs -tail /data/input/customer.csv
```

## 12.6 HDFS validation script

```bash
#!/bin/bash
set -euo pipefail

hdfs_file=${1:-}
expected_count=${2:-}

if [ -z "$hdfs_file" ] || [ -z "$expected_count" ]; then
    echo "Usage: $0 <hdfs_file> <expected_count>"
    exit 1
fi

if ! hdfs dfs -test -e "$hdfs_file"; then
    echo "FAIL: HDFS file does not exist: $hdfs_file"
    exit 1
fi

actual_count=$(hdfs dfs -cat "$hdfs_file" | tail -n +2 | wc -l)

echo "Actual count: $actual_count"
echo "Expected count: $expected_count"

if [ "$actual_count" -eq "$expected_count" ]; then
    echo "PASS: HDFS record count matched"
else
    echo "FAIL: HDFS record count mismatch"
    exit 1
fi
```

## 12.7 Interview explanation

> In Hadoop environments, I use `hdfs dfs` commands to check file arrival, file size, directory contents, and record counts. For data validation, I often pipe `hdfs dfs -cat` output into Unix commands such as `wc`, `awk`, `grep`, `sort`, and `uniq`.

---

# 13. Log Analysis for Batch Jobs

Log analysis is a very common interview area.

## 13.1 Common failure keywords

Search for:

```text
ERROR
FAILED
FAILURE
EXCEPTION
FATAL
ABORTED
REJECTED
PERMISSION DENIED
NO SUCH FILE
CONNECTION REFUSED
TIMEOUT
OUT OF MEMORY
SCHEMA MISMATCH
```

## 13.2 Commands

### Find errors

```bash
grep -Ei "error|exception|failed|fatal|aborted" batch_job.log
```

### Count errors

```bash
grep -Ei "error|exception|failed|fatal|aborted" batch_job.log | wc -l
```

### Find unique errors

```bash
grep -Ei "error|exception|failed|fatal|aborted" batch_job.log | sort | uniq -c | sort -nr
```

### Check recent log lines

```bash
tail -100 batch_job.log
```

### Monitor log while job is running

```bash
tail -f batch_job.log
```

### Show lines around a specific line number

If error is at line 500:

```bash
sed -n '480,520p' batch_job.log
```

## 13.3 Batch failure investigation checklist

When a batch job fails, check:

1. Did the input file arrive?
2. Is the file empty?
3. Is the file in the expected location?
4. Are permissions correct?
5. Did the schema change?
6. Did record count suddenly change?
7. Are there invalid date or numeric values?
8. Did the job fail due to timeout or memory?
9. Did downstream table/file get created?
10. Are rejected records available?

## 13.4 Strong interview answer

> I would start by checking the job log for error keywords using `grep`. Then I would inspect the surrounding log lines to identify the root cause. After that, I would validate input file availability, file size, schema, record count, and rejected records. If it is a downstream issue, I would compare source and target counts and check whether the output was partially generated.

---

# 14. Data Reconciliation Patterns

Data reconciliation is central to Big Data QA.

## 14.1 Count reconciliation

Compare source count and target count.

```bash
source_count=$(tail -n +2 source.csv | wc -l)
target_count=$(tail -n +2 target.csv | wc -l)

if [ "$source_count" -eq "$target_count" ]; then
    echo "PASS"
else
    echo "FAIL"
fi
```

## 14.2 Sum reconciliation

Useful for financial transactions.

```bash
awk -F',' 'NR > 1 {sum += $4} END {printf "%.2f\n", sum}' transactions.csv
```

## 14.3 Duplicate key reconciliation

```bash
awk -F',' 'NR > 1 {print $1}' target.csv | sort | uniq -d
```

## 14.4 Missing key check between source and target

Find keys in source but not target:

```bash
awk -F',' 'NR > 1 {print $1}' source.csv | sort > source_keys.txt
awk -F',' 'NR > 1 {print $1}' target.csv | sort > target_keys.txt

comm -23 source_keys.txt target_keys.txt
```

Find keys in target but not source:

```bash
comm -13 source_keys.txt target_keys.txt
```

## 14.5 Checksum/hash comparison

For full-record comparison:

```bash
sort source.csv | md5sum
sort target.csv | md5sum
```

This is useful when order is not important but full content should match.

## 14.6 Interview explanation

> I would perform reconciliation at multiple levels: count reconciliation, duplicate key checks, mandatory field checks, sum reconciliation for numeric fields, missing key checks, and sample record-level comparison. For financial data, sum reconciliation is very important because counts may match even when values are wrong.

---

# 15. Banking/Data Quality Checks

For RBC or another financial institution, emphasize data accuracy, controls, security, and auditability.

## 15.1 File-level checks

- File exists.
- File is not empty.
- File arrived on time.
- File name follows naming convention.
- File size is within expected range.
- File delimiter is correct.
- File has expected header.
- File has expected trailer, if applicable.
- File date matches processing date.
- File permissions are correct.

## 15.2 Record-level checks

- Required fields are not null.
- Primary keys are unique.
- Data types are valid.
- Date format is valid.
- Numeric fields contain valid numbers.
- Amount fields have correct sign.
- Currency code is valid.
- Account/customer IDs follow expected pattern.
- Status values are from valid domain list.

## 15.3 Reconciliation checks

- Source count equals target count.
- Source amount sum equals target amount sum.
- Source keys exist in target.
- Rejected record count is expected.
- No unexpected duplicates.
- Min/max values are within expected range.
- Control totals match.

## 15.4 Banking-specific examples

- Account number should not be blank.
- Customer ID should not be blank.
- Transaction amount should be numeric.
- Transaction date should not be in an invalid format.
- Currency code should be valid, such as CAD or USD.
- Trade date should not be after processing date unless business rules allow it.
- Sensitive customer data should not be printed unnecessarily in logs.
- Production scripts should generate auditable reports.

## 15.5 Strong RBC-style answer

> Since this is a banking environment, I would make sure validations are repeatable, logged, auditable, and secure. I would avoid printing sensitive customer or account details in logs and would focus on pass/fail summaries, counts, and controlled samples for debugging.

---

# 16. Mock Interview Questions and Strong Answers

## Q1. How do you count records in a file?

```bash
wc -l file.csv
```

If the file has a header:

```bash
tail -n +2 file.csv | wc -l
```

**Answer:**  
I would use `wc -l` to count lines. If the file has a header, I would exclude it using `tail -n +2`.

---

## Q2. How do you check whether a file exists and is not empty?

```bash
if [ -f file.csv ] && [ -s file.csv ]; then
    echo "File exists and is not empty"
else
    echo "File missing or empty"
fi
```

---

## Q3. How do you find duplicate records?

```bash
sort file.csv | uniq -d
```

For duplicate IDs:

```bash
awk -F',' 'NR > 1 {print $1}' file.csv | sort | uniq -d
```

---

## Q4. Difference between grep, awk, and sed?

**Answer:**

- `grep` is mainly for searching text.
- `awk` is for field/column processing and calculations.
- `sed` is for stream editing and text substitution.

Example:

```bash
grep "ERROR" app.log
awk -F',' '{print $1}' file.csv
sed 's/old/new/g' file.txt
```

---

## Q5. How do you check if a batch job failed?

```bash
grep -Ei "error|exception|failed|fatal|aborted" batch_job.log
```

**Answer:**  
I would search the log for failure keywords, check the lines around the error, then validate input files, output files, permissions, schema, and rejected records.

---

## Q6. How do you compare source and target files?

**Answer:**  
I would compare record counts, key counts, duplicate keys, missing keys, numeric totals, and sample records. For large files, I would sort key columns and use `comm` or generate checksums if the full content should match.

---

## Q7. What does `2>&1` mean?

```bash
bash job.sh > job.log 2>&1
```

**Answer:**  
It redirects standard error to the same place as standard output, so both normal messages and error messages go into `job.log`.

---

## Q8. How do you run a script in the background?

```bash
bash script.sh > script.log 2>&1 &
```

---

## Q9. How do you schedule a script?

Using cron:

```bash
crontab -e
```

Example:

```bash
0 2 * * * /home/user/scripts/daily_validation.sh
```

This runs every day at 2:00 AM.

---

## Q10. How do you check disk usage?

```bash
df -h
```

For folder size:

```bash
du -sh /data/input
```

---

## Q11. How do you check running processes?

```bash
ps -ef
```

Find Java processes:

```bash
ps -ef | grep java
```

---

## Q12. How do you validate a file in HDFS?

```bash
hdfs dfs -test -e /data/input/customer.csv
hdfs dfs -cat /data/input/customer.csv | wc -l
hdfs dfs -du -h /data/input/customer.csv
```

**Answer:**  
I would check existence, size, record count, and sample records using `hdfs dfs` commands combined with Unix commands.

---

## Q13. How would you validate a Big Data pipeline?

**Strong answer:**

> I would validate it in layers. First, I would check file arrival, naming convention, file size, and whether the file is empty. Then I would perform source-to-target count validation. After that, I would check schema, column count, nulls, duplicates, data types, date formats, business rules, and rejected records. For large files, I would use Unix commands, HDFS commands, SQL queries, and automated shell scripts. I would also check logs for errors and generate a QA summary report with pass/fail status.

---

## Q14. What makes a shell script production-ready?

**Answer:**  
A production-ready script should validate input arguments, use clear logging, return proper exit codes, quote variables, handle missing files, avoid exposing sensitive data, and be easy to debug. I would also use `set -euo pipefail` where appropriate.

---

## Q15. What is your approach when source and target counts do not match?

**Answer:**

1. Confirm source and target queries/files are correct.
2. Check whether headers/trailers were included accidentally.
3. Check rejected records.
4. Check filters or business rules.
5. Check duplicate keys.
6. Check job logs for errors.
7. Identify missing keys using `comm`, SQL joins, or reconciliation queries.
8. Report the mismatch with evidence.

---

# 17. Hands-On Coding Exercises

Use these exercises for practice.

---

## Exercise 1: Count records in all CSV files

```bash
#!/bin/bash
set -euo pipefail

for file in *.csv
do
    count=$(tail -n +2 "$file" | wc -l)
    echo "$file : $count"
done
```

---

## Exercise 2: Find files modified in the last 24 hours

```bash
find /data/input -type f -mtime -1
```

---

## Exercise 3: Find files older than 7 days

```bash
find /data/archive -type f -mtime +7
```

---

## Exercise 4: Archive old log files

```bash
#!/bin/bash
set -euo pipefail

archive_dir="/data/archive"
log_dir="/data/logs"

mkdir -p "$archive_dir"

find "$log_dir" -name "*.log" -mtime +7 -exec gzip {} \;
find "$log_dir" -name "*.log.gz" -mtime +7 -exec mv {} "$archive_dir" \;

echo "Old logs archived"
```

---

## Exercise 5: Validate mandatory fields

```bash
awk -F',' 'NR > 1 && ($1 == "" || $4 == "") {print NR, $0}' customer.csv
```

---

## Exercise 6: Generate duplicate report

```bash
#!/bin/bash
set -euo pipefail

file=${1:-}
report="duplicate_customer_ids.txt"

awk -F',' 'NR > 1 {print $1}' "$file" | sort | uniq -d > "$report"

echo "Duplicate report generated: $report"
cat "$report"
```

---

## Exercise 7: Compare keys between two files

```bash
#!/bin/bash
set -euo pipefail

source_file=${1:-}
target_file=${2:-}

awk -F',' 'NR > 1 {print $1}' "$source_file" | sort > source_keys.txt
awk -F',' 'NR > 1 {print $1}' "$target_file" | sort > target_keys.txt

echo "Keys in source but missing in target:"
comm -23 source_keys.txt target_keys.txt

echo "Keys in target but missing in source:"
comm -13 source_keys.txt target_keys.txt
```

---

# 18. Mini Project: QA Validation Framework

This mini project is useful to discuss in an interview.

## 18.1 Goal

Create a reusable shell script that validates a data file and generates a QA report.

## 18.2 Requirements

The script should check:

- File exists.
- File is not empty.
- Record count excluding header.
- Column count.
- Duplicate key count.
- Null count for mandatory columns.
- Invalid date format count.
- Amount sum.
- Pass/fail status.

## 18.3 Example input file

```text
customer_id,name,email,status,amount,transaction_date
1001,Amit,amit@example.com,ACTIVE,250.50,2026-05-01
1002,Sara,sara@example.com,ACTIVE,100.00,2026-05-01
1003,John,,INACTIVE,75.25,2026-05-02
1002,Sara,sara@example.com,ACTIVE,100.00,2026-05-01
```

## 18.4 Framework script

```bash
#!/bin/bash
set -euo pipefail

input_file=${1:-}
expected_columns=${2:-6}
report="qa_report_$(date '+%Y%m%d_%H%M%S').txt"

log() {
    echo "$1" | tee -a "$report"
}

fail_flag=0

if [ -z "$input_file" ]; then
    echo "Usage: $0 <input_file> [expected_columns]"
    exit 1
fi

log "QA Validation Report"
log "===================="
log "File: $input_file"
log "Generated At: $(date)"
log ""

if [ ! -f "$input_file" ]; then
    log "FAIL: File does not exist"
    exit 1
fi

if [ ! -s "$input_file" ]; then
    log "FAIL: File is empty"
    exit 1
fi

record_count=$(tail -n +2 "$input_file" | wc -l)
invalid_col_count=$(awk -F',' -v cols="$expected_columns" 'NR > 1 && NF != cols {count++} END {print count+0}' "$input_file")
duplicate_count=$(awk -F',' 'NR > 1 {print $1}' "$input_file" | sort | uniq -d | wc -l)
blank_customer_id_count=$(awk -F',' 'NR > 1 && $1 == "" {count++} END {print count+0}' "$input_file")
blank_status_count=$(awk -F',' 'NR > 1 && $4 == "" {count++} END {print count+0}' "$input_file")
invalid_date_count=$(awk -F',' 'NR > 1 && $6 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ {count++} END {print count+0}' "$input_file")
amount_sum=$(awk -F',' 'NR > 1 {sum += $5} END {printf "%.2f", sum}' "$input_file")

log "Record Count Excluding Header: $record_count"
log "Invalid Column Count Rows: $invalid_col_count"
log "Duplicate Customer IDs: $duplicate_count"
log "Blank Customer IDs: $blank_customer_id_count"
log "Blank Status Values: $blank_status_count"
log "Invalid Transaction Dates: $invalid_date_count"
log "Total Amount Sum: $amount_sum"
log ""

if [ "$invalid_col_count" -gt 0 ]; then
    fail_flag=1
fi

if [ "$duplicate_count" -gt 0 ]; then
    fail_flag=1
fi

if [ "$blank_customer_id_count" -gt 0 ]; then
    fail_flag=1
fi

if [ "$blank_status_count" -gt 0 ]; then
    fail_flag=1
fi

if [ "$invalid_date_count" -gt 0 ]; then
    fail_flag=1
fi

if [ "$fail_flag" -eq 0 ]; then
    log "OVERALL STATUS: PASS"
    exit 0
else
    log "OVERALL STATUS: FAIL"
    exit 1
fi
```

## 18.5 How to explain this project in an interview

> I created a reusable shell-based QA validation framework that accepts a file and expected column count as arguments. It validates file existence, empty file condition, record count, column count, duplicates, mandatory fields, date format, and numeric totals. It generates a timestamped QA report and returns a proper exit code so it can be integrated with a scheduler or CI/CD pipeline.

---

# 19. 7-Day Preparation Plan

## Day 1: Unix basics

Practice:

```bash
pwd
ls -ltr
cd
mkdir
cp
mv
rm
touch
chmod
```

Goal: Be comfortable navigating directories and managing files.

---

## Day 2: File inspection and logs

Practice:

```bash
cat
less
head
tail
tail -f
wc -l
grep
grep -i
grep -E
```

Goal: Be able to inspect files and search logs.

---

## Day 3: sort, uniq, cut, awk

Practice:

```bash
sort
uniq
uniq -c
uniq -d
cut -d',' -f1
awk -F',' '{print $1}'
awk -F',' 'NF != 5'
```

Goal: Be able to validate columns, duplicates, and counts.

---

## Day 4: Shell scripting

Practice:

- Variables
- Arguments
- If conditions
- Loops
- Functions
- Exit codes
- Logging

Goal: Write simple validation scripts without looking up every syntax detail.

---

## Day 5: Big Data QA scripts

Build scripts for:

- File existence check
- Empty file check
- Record count validation
- Column count validation
- Duplicate check
- Null check
- Date format check

Goal: Connect Unix scripting to QA validation scenarios.

---

## Day 6: HDFS/Hadoop commands

Practice:

```bash
hdfs dfs -ls
hdfs dfs -cat
hdfs dfs -head
hdfs dfs -tail
hdfs dfs -count
hdfs dfs -du -h
hdfs dfs -test -e
hdfs dfs -put
hdfs dfs -get
```

Goal: Be ready for Hadoop-based data pipeline questions.

---

## Day 7: Mock interview

Practice answering:

- How do you validate a Big Data pipeline?
- How do you debug a failed batch job?
- How do you compare source and target counts?
- How do you find duplicates?
- How do you validate HDFS files?
- How do you make a shell script production-ready?

Goal: Explain your approach clearly and confidently.

---

# 20. Final Cheat Sheet

## 20.1 Must-know Unix commands

```bash
pwd
ls -ltr
cd
mkdir
cp
mv
rm
touch
chmod
cat
less
head
tail
tail -f
wc -l
grep -i
grep -E
find
sort
uniq
uniq -c
uniq -d
cut
awk
sed
df -h
du -sh
ps -ef
crontab
```

## 20.2 Must-know HDFS commands

```bash
hdfs dfs -ls
hdfs dfs -cat
hdfs dfs -head
hdfs dfs -tail
hdfs dfs -count
hdfs dfs -du -h
hdfs dfs -test -e
hdfs dfs -put
hdfs dfs -get
hdfs dfs -rm
```

## 20.3 Must-know QA validations

```text
File arrival validation
File size validation
Empty file validation
Header/trailer validation
Record count validation
Column count validation
Null validation
Duplicate validation
Date format validation
Numeric format validation
Source-to-target reconciliation
Sum reconciliation
Missing key validation
Rejected record validation
Log error validation
```

## 20.4 Must-know scripting concepts

```text
Shebang
Variables
Arguments
If conditions
For loops
While loops
Case statements
Functions
Exit codes
Logging
Cron scheduling
Input validation
Quoting variables
set -euo pipefail
```

## 20.5 Best final interview pitch

> My Unix scripting strength is in automating Big Data QA checks. I can validate file arrival, record counts, column counts, nulls, duplicates, date formats, amount totals, and source-to-target reconciliation. I can also analyze batch logs, work with HDFS files, and create pass/fail QA reports with proper logging and exit codes. In a banking environment, I would make sure the checks are repeatable, auditable, and do not expose sensitive customer information.

---

# 21. References

These references are useful for deeper study and command verification:

1. RBC Careers, Sr. Quality Engineer postings mentioning Unix/Linux, Big Data/Hadoop, cloud, automated testing, SQL, Python, and QA/test environments:  
   - https://jobs.rbc.com/ca/en/job/R-0000172211/Sr-Quality-Engineer  
   - https://jobs.rbc.com/ca/en/job/R-0000166737/Senior-Quality-Engineer-Halifax

2. GNU Bash Manual:  
   - https://www.gnu.org/software/bash/manual/bash.html

3. GNU Awk User's Guide:  
   - https://www.gnu.org/software/gawk/manual/gawk.html

4. Apache Hadoop FileSystem Shell Documentation:  
   - https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html

---

## Final Note

For an interview, do not just say commands. Connect each command to a QA scenario.

Weak answer:

> I know grep, awk, sed, and HDFS commands.

Strong answer:

> I use `grep` for log analysis, `awk` for column-level data validation, `sort` and `uniq` for duplicate checks, `wc -l` for record counts, and `hdfs dfs` commands to validate files in Hadoop. I combine these into shell scripts that generate repeatable QA reports with pass/fail status.

