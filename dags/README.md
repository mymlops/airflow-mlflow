# Apache Airflow DAGs

Place all your Airflow DAGs in this folder.

Airflow requires that all DAGs be placed in a default folder configured in `~/.airflow/airflow.cfg`. This has already been configured for you.

# Registering DAGs

To register a DAG with Airflow, `cd` into this directory and run the command below 

```
python tutorial.py
```

Head over to Airflow UI and you should see a DAG named `Tutorial`.

> It may take a while for Airflow to recognize a new DAG.