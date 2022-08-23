"""Script for preparing the current loanbook size report."""

import pandas as pd
import numpy as np

from utils.utils import LoanBook
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta

loanbook = LoanBook()

### Start of DAGs
with DAG(
    dag_id="loanbooksize_pipeline",
    description="Prepares the current loanbook size report.",
    schedule_interval=timedelta(hours=4),
    default_args=loanbook.default_args,
    catchup=False,
) as dag:

    # Task 1: read loans table
    read_loans_table = PythonOperator(
        task_id="read_loans_table",
        python_callable=loanbook.read_bq_table,
        op_kwargs={
            "tablename": "loans_view",
            "datasetid": "firestore_loans",
            "projectid": "payhippo",
        },
    )

    # TODO: continue from here
    # Task 2: read repaymentSchedules table
    read_repaymentSchedules_table = PythonOperator(
        task_id="read_repaymentSchedules_table",
        python_callable=loanbook.read_bq_table,
        op_kwargs={
            "tablename": "repaymentSchedules_view",
            "datasetid": "firestore_repaymentSchedules",
            "projectid": "payhippo",
        },
    )

    # Task 3: compile loan collections
    get_loan_collections = PythonOperator(
        task_id="get_loan_collections",
        python_callable=loanbook.get_loan_collections,
        op_kwargs={
            "repaymentSchedules": "repaymentSchedules"
        },  # TODO: pass a dataframe
    )

    # Task 4: process loans
    process_loans = PythonOperator(
        task_id="process_loans",
        python_callable=loanbook.process_loans,
        op_kwargs={
            "loans": "loans",
            "loan_collections": "loan_collections",  # TODO: pass a dataframe
        },
    )


outstanding_loans = loans_proc[loanbook.analysis_cols].query(
    "isRepaid == False"
)

active_outstanding_loans = (
    outstanding_loans.groupby(by=["disbursed_ym", "category_90"])
    .aggregate(
        loanAmount=("loanAmount", "sum"),
        loan_count=("loanId", "count"),
    )
    .unstack("category_90")
    .fillna(0)
    .assign(
        outstanding_loanAmount=lambda df: df[("loanAmount", "0")]
        + df[("loanAmount", "1-90")]
        + df[("loanAmount", ">90")],
        outstanding_loan_count=lambda df: df[("loan_count", "0")]
        + df[("loan_count", "1-90")]
        + df[("loan_count", ">90")],
        active_loanAmount=lambda df: df[("loanAmount", "0")]
        + df[("loanAmount", "1-90")],
        active_loan_count=lambda df: df[("loan_count", "0")]
        + df[("loan_count", "1-90")],
    )[
        [
            "outstanding_loanAmount",
            "outstanding_loan_count",
            "active_loanAmount",
            "active_loan_count",
        ]
    ]
    .reset_index()
)

active_outstanding_loans = active_outstanding_loans.droplevel(1, axis=1)

# Unit tests
assert (
    outstanding_loans.loanId.count()
    == active_outstanding_loans.outstanding_loan_count.sum()
)
assert (
    outstanding_loans.loanAmount.sum()
    == active_outstanding_loans.outstanding_loanAmount.sum()
)

current_loan_book = (
    loans_proc.groupby("disbursed_ym")
    .aggregate(
        loanAmount=("loanAmount", "sum"),
    )
    .reset_index()
    .merge(right=active_outstanding_loans, how="left", on="disbursed_ym")
    .fillna(0)
)

current_loan_book = current_loan_book.rename(columns=loanbook.col_names)
