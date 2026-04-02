
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

# Q1 Naive Bayes

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    tokenized_texts = [text.split() for text in texts]

    vocabulary = set()
    for tokens in tokenized_texts:
        vocabulary.update(tokens)

    total_docs = len(labels)
    priors = {
        1: np.sum(labels == 1) / total_docs,
        0: np.sum(labels == 0) / total_docs
    }

    word_probs = {0: {}, 1: {}}

    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for tokens, label in zip(tokenized_texts, labels):
        for word in tokens:
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    for c in [0, 1]:
        for word in vocabulary:
            count = word_counts[c].get(word, 0)
            if total_words[c] > 0:
                word_probs[c][word] = count / total_words[c]
            else:
                word_probs[c][word] = 0

    test_tokens = test_email.split()

    scores = {}
    for c in [0, 1]:
        score = priors[c]
        for word in test_tokens:
            prob = word_probs[c].get(word, 0)
            score *= prob  
        scores[c] = score

    prediction = 1 if scores[1] > scores[0] else 0

    return priors, word_probs, prediction

# Q2 KNN

def knn_iris(k=3, test_size=0.2, seed=0):

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(X_train, y_train, X_test):
        predictions = []

        for test_point in X_test:
            distances = []

            for x_train, label in zip(X_train, y_train):
                dist = euclidean_distance(test_point, x_train)
                distances.append((dist, label))

            distances.sort(key=lambda x: x[0])

            k_neighbors = [label for _, label in distances[:k]]

            counts = np.bincount(k_neighbors)
            pred_label = np.argmax(counts)

            predictions.append(pred_label)

        return np.array(predictions)

    train_predictions = predict(X_train, y_train, X_train)
    test_predictions = predict(X_train, y_train, X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
