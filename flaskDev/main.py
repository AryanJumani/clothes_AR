from flask import Flask, request

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def API():
    Query = str(request.args['Query'])
    return Query
def backendPreprocess(Query):
    return Query

