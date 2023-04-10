import joblib
from ..preprocess.text import dirty_to_clean

SVM_MODELS_PATH = "core/predictors/svm/svm_models.joblib"

svm_models = joblib.load(SVM_MODELS_PATH)


def predict_svm(title, text):
    question = title + " " + text
    X = dirty_to_clean([question])
    topics = [
        "array",
        "string",
        "dynamic_programming",
        "math",
        "hash_table",
        "greedy",
        "sorting",
        "depth_first_search",
        "breadth_first_search",
        "binary_search",
    ]

    topic_mapping = {
        "array": 0,
        "string": 1,
        "dynamic_programming": 2,
        "math": 3,
        "hash_table": 4,
        "greedy": 5,
        "sorting": 6,
        "depth_first_search": 7,
        "breadth_first_search": 8,
        "binary_search": 9,
    }

    res = {}
    for topic in topics:
        prediction = svm_models[topic_mapping[topic]].predict_proba(X)
        res.update({topic: prediction[0][1]})
    return res
