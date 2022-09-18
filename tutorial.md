# Experimentation flow

We start by exploring and understanding the data. Next, we test three different embeddings: bag-of-words, TF-IDF, and Word2vec, using the K-nearest neighbors (KNN) classifier to train the models. Our goal is to select the best model to take to production.

## Edit the notebook

The experimentation flow starts from the notebook and your data. Version the dataset with DVC and the code with Git before running the notebook to create reproducible experiments. Doing so allows MLflow to track the data and code used during training and store this information together with your experiment results.

## Version the data

To start versioning the data, you must tell DVC which files to version. Do this by using the command below, specifying the data files, and pushing the changes to the DVC repository.

```bash
dvc add data/train.csv
dvc push
```

DVC will generate a human-readable `.dvc` file. This file is a data placeholder and must be committed with your code using Git. Your actual data files will be included in `.gitignore` and will not be pushed to your Git repo, keeping the data and code layers separate.

## Version the code

Use Git to stage the `.dvc` file and any modifications made to the notebook or any other relevant code in the project.

```bash
git add data/train.csv.dvc notebooks/ToxicCommentClassifier.ipynb
```

Finally, commit the changes.

```bash
git commit -m 'Comment about the code and data'
```

After completing these steps, you will effectively create a snapshot of your data and code. You can use the Git commit sha to retrieve that snapshot anytime. Optionally, you can mark specific points in your repo's history using Git tags to retrieve them more easily.

## Track metrics and artifacts

Use MLflow to track metrics from your experiments. We use the `autolog` function to track most metrics and artifacts automatically.

```python
mlflow.sklearn.autolog()
```

You can also manually log additional parameters. For example, we manually associate the Git commit sha with the current MLflow run.

```python
mlflow.set_tag('mlflow.source.git.commit', sha_commit)
```

This means you can go back in time and see the exact data and code that produced those results.

## Run the notebook and iterate

Run your notebook and check your results in MLflow. Rinse and repeat. Make a change to the code or data, then use DVC and Git to version the changes. When you rerun your experiment, MLflow will track and associate your results with the data and code versions you used. Over time, you will have a list of experiments in MLflow. You can reproduce any of them and pick the best one.

In our tutorial, the best results were produced by the Word2vec embedding. We select this model to take to production in the following steps.

# Production flow

In the production workflow, we create a pipeline to automate data and code versioning, data preprocessing, and finally, model retraining and serving.

## Create a pipeline

The pipeline is created using Airflow and defined in a `.py` file. A pipeline is also known as a Direct Acyclic Graph (DAG). It automates all necessary steps to go from data to a deployed model. Each pipeline run will retrain and redeploy the model based on the latest state of your dataset. Airflow triggers the pipeline at a fixed schedule, assuming your application regularly collects new data between the triggers.

## Add data versioning step

Similarly to the experimentation flow, you want to automatically version your dataset with DVC, so you know which data produced that model. In Airflow, we use the `BashOperator` to execute the same DVC versioning commands we used during experimentation.

## Add code versioning step

The next pipeline step uses the `BashOperator` to version the `.dvc` file and other code changes using Git commands. This creates a snapshot of your data and code's current state, enabling the reproduction of the results.

## Add training step

We'll manually copy and refactor the preprocessing and training code from our notebook and add it as a step in the pipeline. Once again, we use MLflow to track training metrics and artifacts. We also use MLflow's model registry to track which model versions are in staging and in production. Promoting a model to production in MLflow does not mean the model is actually deployed. The information in MLflow's registry is just for logging purposes, and the deployment is independent of MLflow.

## Add model deployment steps

We use BentoML to build and serve the models. BentoML packages the model by importing the relevant files from MLflow and creates a Docker container to run it in production. You must create a `.py` file defining the service endpoints and a `bentofile.yaml` configuration file. We use Airflow to bring the container online with Docker Compose, configured through a `.yml` file.

## Export the model serving metrics

BentoML exports serving metrics out-of-the-box. We configure Prometheus to regularly collect these metrics. You can also define your own custom metrics if needed.

## Monitor the model

The metrics collected by Prometheus are used to create monitoring dashboards with Grafana. An example dashboard is provided as a `.json` file. This can be imported and customized. You can also use Grafana to create alerts based on the metrics to warn you when something is not working correctly.

## Make predictions

Finally, we test our setup using a notebook to make prediction requests to the deployed model. After sending some requests, you can visualize the metrics in Grafana. The prediction requests in a typical production workflow would come from your own application.
