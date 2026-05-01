from flask import Flask, request, jsonify, send_file
from assistant import handle_query, run_vmec
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/run_vmec_ui", methods = ["POST"])
def run_vmec_ui():
    data = request.json
    ns = data.get("ns")
    mpol = data.get("mpol")
    ntor = data.get("ntor")
    result = run_vmec(ns, mpol, ntor)
    return {"result": result}
@app.route("/plot/<filename>")
def plot(filename):
    return send_file(filename)

@app.route("/")
def home():
    return send_file("index.html")
@app.route("/chat", methods = ["POST"])
def chat():
    question = request.json.get('message', '')
    answer = handle_query(question)
    return  jsonify({'response' : answer})
if __name__ == "__main__":
    app.run(debug = True, port = 5001)

