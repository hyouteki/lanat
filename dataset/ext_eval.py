#Libraries

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

#Training

#Features

with open('corpus.txt', 'r') as file:
    corpus_train = file.readlines()
for i in range(len(corpus_train)):
    corpus_train[i]=corpus_train[i].strip()


#Cat_Var

with open('labels.txt', 'r') as file:
    labels_train = file.readlines()
for i in range(len(labels_train)):
    labels_train[i]=labels_train[i].strip()

#Testint

with open('lm_corpus.txt', 'r') as file:
    corpus_test = file.readlines()
for i in range(len(corpus_test)):
    corpus_test[i]=corpus_test[i].strip()


#Cat_Var

with open('lm_labels.txt', 'r') as file:
    labels_test = file.readlines()
for i in range(len(labels_test)):
    labels_test[i]=labels_test[i].strip()


#Vectorizer

vectorizer = TfidfVectorizer()
corpus_train_tfidf = vectorizer.fit_transform(corpus_train)
corpus_test_tfidf = vectorizer.transform(corpus_test)


#SVC

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
model.fit(corpus_train_tfidf, labels_train)

# Predictions
predictions = model.predict(corpus_test_tfidf)

# Evaluate the accuracy
accuracy = accuracy_score(labels_test, predictions)
print(f"Accuracy: Before Tuning: {accuracy}")

#Tuning

param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(corpus_train_tfidf, labels_train)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Use the best model for predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(corpus_test_tfidf)

# Evaluate the accuracy
accuracy = accuracy_score(labels_test, predictions)
print(f"Accuracy: After Tuning {accuracy}")

#Evaluate the F1 scores
macro_f1 = f1_score(labels_test, predictions, average='macro')
print(f"Final Macro F1 Score: {macro_f1}")
