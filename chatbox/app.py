from flask import Flask , render_template , request , jsonify
from src.emobotmain import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/reply" , methods=['POST'])
def reply():
    text = request.json['text']
    return jsonify(dict(text=predict(text)))


if __name__=="__main__":
    app.run(debug = True)
