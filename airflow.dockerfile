FROM apache/airflow:2.9.1

USER airflow
# Only DVC needed — project venv is mounted at /opt/airflow/project/venv
RUN pip install --no-cache-dir dvc==3.67.1
