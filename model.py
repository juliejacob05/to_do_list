from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Training data
tasks = ["Buy groceries", "Finish project report", "Call the doctor", "Clean room", "Deploy the app", "Visit the doctor", "Go to the hospital for appointment", "Shopping", "Paint the room"]
labels = ["Personal", "Work", "Health", "Personal", "Work", "Health", "Health", "Personal", "Personal"]

# Vectorize and train
vec = CountVectorizer()
X = vec.fit_transform(tasks)

model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)
