from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

tasks = ["Buy groceries", "Finish project report", "Call the doctor", "Clean room", "Deploy the app", "Visit the doctor", "Go to the hospital for appointment", "Complete the project", "Paint the room"]
labels = ["Personal", "Work", "Health", "Personal", "Work", "Health", "Health", "Work", "Personal"]

# Vectorize text
vec = CountVectorizer()
X = vec.fit_transform(tasks)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict
new_task = ["Checkup at hospital"]
X_new = vec.transform(new_task)
print("Predicted:", model.predict(X_new))
