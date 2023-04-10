import joblib
from ..preprocess.text import dirty_to_clean

XGB_MODELS_PATH = "core/predictors/xgb/xgb_models.joblib"

clf = joblib.load(XGB_MODELS_PATH)


def predict_xgb(title, text):
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
        "array": "Array",
        "string": "String",
        "dynamic_programming": "Dynamic Programming",
        "math": "Math",
        "hash_table": "Hash Table",
        "greedy": "Greedy",
        "sorting": "Sorting",
        "depth_first_search": "Depth-First Search",
        "breadth_first_search": "Breadth-First Search",
        "binary_search": "Binary Search",
    }

    res = {}
    for topic in topics:
        prediction = clf[topic_mapping[topic]].predict_proba(X)
        res.update({topic: prediction[0][1]})
    return res
