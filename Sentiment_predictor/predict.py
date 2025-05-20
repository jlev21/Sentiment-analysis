import joblib
import numpy as np
def score_to_prob(score):
    return 1 / (1 + np.exp(-score))

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('svm_model.pkl')

# Get user input
review = input("Enter a review: ")

# Vectorize the input text
X_input = vectorizer.transform([review])

# Predict class
pred_class = model.predict(X_input)[0]
if pred_class == 0:
    pred_class = "negative"
else:
    pred_class = "positive"
# Get decision score (confidence estimate)
score = score_to_prob(model.decision_function(X_input)[0])

# Display prediction
print("\nPrediction Result:")
print(f"Predicted class: {pred_class}")
print(f"Probability: {score:.4f}%")
