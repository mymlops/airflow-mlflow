from datetime import datetime, timedelta
from os.path import dirname, abspath
import os
import sys
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator, get_current_context

tutorial_dir_path = dirname(dirname(abspath(__file__)))


def load_train_test_data(**kwargs):
    import os
    import dvc.api
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_path = os.path.join(tutorial_dir_path, "data/train.csv")

    data_url = dvc.api.get_url(path=data_path)

    data = pd.read_csv(data_url, nrows=5000)

    X = data['comment_text']
    y = data.iloc[:, 2:]

    return train_test_split(X, y, test_size=0.2)


def create_model(**kwargs):
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    class TextPreprocessor(BaseEstimator):
        """TextPreprocessor preprocesses text by applying these rules:

        - Remove special chars
        - Remove punctuation
        - Convert to lowercase
        - Replace numbers
        - Tokenize text
        - Remove stopwords
        - Lemmatize words

        It implements the BaseEstimator interface and can be used in sklearn pipelines.
        """

        def remove_special_chars(self, text):
            import re
            import html

            re1 = re.compile(r'  +')
            x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace('\\', ' \\ ')
            return re1.sub(' ', html.unescape(x1))

        def remove_punctuation(self, text):
            """Remove punctuation from list of tokenized words"""
            import string

            translator = str.maketrans('', '', string.punctuation)
            return text.translate(translator)

        def to_lowercase(self, text):
            return text.lower()

        def replace_numbers(self, text):
            """Replace all interger occurrences in list of tokenized words with textual representation"""
            import re

            return re.sub(r'\d+', '', text)

        def text2words(self, text):
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)

        def remove_stopwords(self, words):
            """
            :param words:
            :type words:
            :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
            or
            from spacy.lang.en.stop_words import STOP_WORDS
            :type stop_words:
            :return:
            :rtype:
            """
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')

            return [word for word in words if word not in stop_words]

        def lemmatize_words(self, words):
            """Lemmatize words in text"""
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(word) for word in words]

        def lemmatize_verbs(self, words):
            """Lemmatize verbs in text"""
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

        def clean_text(self, text):
            text = self.remove_special_chars(text)
            text = self.remove_punctuation(text)
            text = self.to_lowercase(text)
            text = self.replace_numbers(text)
            words = self.text2words(text)
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            words = self.lemmatize_verbs(words)

            return ''.join(words)

        def fit(self, x, y=None):
            return self

        def transform(self, x):
            return map(lambda x: self.clean_text(x), x)

    class Word2vecVectorizer(BaseEstimator):
        """ Wor2vecVectorizer provides a wrapper around gensim's Word2Vec model
        to be used in sklearn's pipeline.
        """

        def __init__(self, min_count=5, vector_size=100, window=5):
            self.min_count = min_count
            self.vector_size = vector_size
            self.window = window
            self.sentences = []

        def fit(self, x, y=None):
            from gensim.models import Word2Vec

            self.sentences = list(map(lambda k: k.split(), x))

            self.model = Word2Vec(
                min_count=self.min_count, vector_size=self.vector_size, window=self.window, sg=1)
            self.model.build_vocab(self.sentences, progress_per=10000)
            self.model.train(
                self.sentences, total_examples=self.model.corpus_count, epochs=20)

            return self

        def transform(self, x):
            import numpy as np

            sentences = list(map(lambda k: k.split(), x))
            if len(sentences) != 0:
                self.sentences = sentences
            w2v_words = list(self.model.wv.index_to_key)

            # We calculate the sentence embedding as the average of the embedding of the words in the sentence
            vector = []
            for sentence in self.sentences:
                sentence_vec = np.zeros(self.vector_size)
                count = 0
                for word in sentence:
                    if word in w2v_words:
                        vec = self.model.wv[word]
                        sentence_vec += vec
                        count += 1
                if count != 0:
                    sentence_vec /= count  # averaging
                vector.append(sentence_vec)

            return vector

    return Pipeline([
        ('preprocessing', TextPreprocessor()),
        ('word2vec', Word2vecVectorizer()),
        ('knn', KNeighborsClassifier(n_neighbors=6))
    ])


def _train_model():
    import mlflow
    import bentoml
    from mlflow.tracking import MlflowClient
    import git
    repo = git.Repo(tutorial_dir_path, search_parent_directories=True)
    sha_commit = repo.head.object.hexsha

    x_train, x_test, y_train, y_test = load_train_test_data()

    model = create_model()

    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.source.git.commit', sha_commit)

        model.fit(x_train, y_train)
        model.score(x_test, y_test)

        extra_pip_requirements = ["nltk", "numpy"]

        mlflow.sklearn.log_model(
            model, "model", extra_pip_requirements=extra_pip_requirements)

        registered_model = mlflow.register_model(
            "runs:/{}/model".format(run.info.run_id), "ToxicCommentClassifier")

        bentoml.mlflow.import_model(
            "toxic-comment-classifier",
            registered_model.source,
            signatures={"predict": {"batchable": True}},
        )

    client = MlflowClient()

    client.transition_model_version_stage(
        name="ToxicCommentClassifier",
        version=registered_model.version,
        stage="Production",
    )

    context = get_current_context()

    context["ti"].xcom_push(key='model_uri', value=registered_model.source)

    return True


with DAG(
    'toxic_comment_classifier',
    description="Pipeline for training and deploying a classifier of toxic comments",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["tutorial"]
) as dag:
    import os
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Toxic Comment Classifier")
    mlflow.sklearn.autolog(silent=True, log_models=False)

    # We start by versioning our data and code to make sure our results can be traced back to the data and code that generated it.
    # We assume the latest data has been loaded to the .csv files under the data folder.
    # Our data does not change, but in a practical scenario, the data could have changed since last run.
    t1 = BashOperator(
        task_id="version_data",
        bash_command="cd {} && dvc add data/train.csv && dvc push".format(
            tutorial_dir_path)
    )

    # We version all code files using Git. The data file 'train.csv' won't be commited
    # because it was added to .gitignore by DVC
    t2 = BashOperator(
        task_id="stage_files",
        bash_command="cd {} && git add .".format(tutorial_dir_path)
    )

    t3 = BashOperator(
        task_id="commit_files",
        bash_command="cd {} && git commit -m 'Update data file' || echo 'No changes to commit'".format(
            tutorial_dir_path)
    )

    t4 = ShortCircuitOperator(
        task_id="train_model",
        python_callable=_train_model
    )

    bento_path = os.path.join(tutorial_dir_path, "include/bentoml")

    # We export BENTOML_MLFLOW_MODEL_PATH to be used in the bentofile.yaml so BentoML can find the model's requirements.txt
    t5 = BashOperator(
        task_id="build_model",
        bash_command="cd {} && export BENTOML_MLFLOW_MODEL_PATH={{{{ ti.xcom_pull(key='model_uri') }}}} && bentoml build".format(
            bento_path)
    )

    t6 = BashOperator(
        task_id="containerize_model",
        bash_command="bentoml containerize toxic-comment-classifier:latest -t toxic-comment-classifier:latest"
    )

    docker_compose_file_path = os.path.join(
        tutorial_dir_path, "include/docker-compose/docker-compose.yml")

    t7 = BashOperator(
        task_id="serve_model",
        bash_command="docker compose -f {} up -d --wait".format(
            docker_compose_file_path)
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
