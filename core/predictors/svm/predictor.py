import joblib
from ..preprocess.text import clean_questions, number_count_questions, embed, combine_number_count

XGB_MODELS_PATH = "core/predictors/svm/svm_models.joblib"

svm_models = joblib.load(SVM_MODELS_PATH)

def dirty_to_clean(questions):
    """
    Function to clean the questions
    Input: qns: list of strings
    Output: doc_clean - list of strings
    """
    cleaned_questions = clean_questions(questions)
    number_count = number_count_questions(questions)
    embed_questions = embed(cleaned_questions, embed_type="glove")
    tmp = np.array(embed_questions)
    return combine_number_count(embed_questions, number_count)

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
