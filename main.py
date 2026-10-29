from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# This example demonstrates a very basic form of Artificial Intelligence:
# a text classifier that can learn to distinguish between two categories of text.
# It uses a simple machine learning model (Naive Bayes) and text vectorization.

# Sample training data: emails labeled as 'spam' or 'not_spam'
training_data = [
    ("Great news! You've won a free vacation! Click here to claim.", "spam"),
    ("Meeting reminder: Project X sync at 3 PM tomorrow.", "not_spam"),
    ("Limited time offer! Get 50% off on all electronics.", "spam"),
    ("Your order has been shipped. Tracking number: 12345.", "not_spam"),
    ("Congratulations! You're selected for a special prize. Call now!", "spam"),
    ("Regarding your recent inquiry, please find the attached document.", "not_spam")
]

# Separate texts and labels
texts = [data[0] for data in training_data]
labels = [data[1] for data in training_data]

# Create a machine learning pipeline:
# 1. CountVectorizer: Converts text into a matrix of token counts (features).
# 2. MultinomialNB: A simple probabilistic classifier suitable for text classification.
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model using the training data
model.fit(texts, labels)

# --- Now, let's test the trained model ---

print("--- AI Text Classifier Demo ---")
print("Model trained on spam/not_spam emails.")
print("------------------------------\n")

def predict_email_type(email_text):
    # Predict the category of a new email
    prediction = model.predict([email_text])
    return prediction[0]

# Test with new, unseen emails

email1 = "Claim your prize now! Limited time offer!"
prediction1 = predict_email_type(email1)
print(f"Email: '{email1}'\nPrediction: {prediction1}\n")

email2 = "Hi, just a quick update on the project status."
prediction2 = predict_email_type(email2)
print(f"Email: '{email2}'\nPrediction: {prediction2}\n")

email3 = "You have won a lottery! Call this number immediately."
prediction3 = predict_email_type(email3)
print(f"Email: '{email3}'\nPrediction: {prediction3}\n")

email4 = "Your appointment is scheduled for next Tuesday."
prediction4 = predict_email_type(email4)
print(f"Email: '{email4}'\nPrediction: {prediction4}\n")
