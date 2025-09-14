import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
import json

# 1) Load intents.json
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

# 2) Prepare training data
texts = []   # patterns
labels = []  # tags

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# 3) Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 4) Train model (Logistic Regression – fast)
model = LogisticRegression()
model.fit(X, y)

print("✅ Chatbot Training Complete! Type 'quit' to exit.\n")

# 5) Chat function
def chatbot_response(user_input):
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]

    # find matching response
    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

# 6) Chat loop
while True:
    user = input("You: ")
    if user.lower() == "quit":
        print("Bot: Bye! Have a nice day 👋")
        break
    print("Bot:", chatbot_response(user))
