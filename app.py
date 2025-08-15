from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

keyword_responses = {
    "hi": "Hello there! 😊",
    "hello": "Hi! How can I assist you today?",
    "hey": "Hey! What's up?",
    "bye": "Goodbye! Have a great day! 👋",
    "see you": "See you soon!",
    "thanks": "You're welcome! 😊",
    "thank you": "Anytime! Glad I could help.",
    "help": "I'm here to help. What do you need?",
}

default_response = "I'm not sure how to respond to that, but I'm here for you."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").lower().strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    reply = default_response
    for keyword in keyword_responses:
        if keyword in user_input:
            reply = keyword_responses[keyword]
            break

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5006)

