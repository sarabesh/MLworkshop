import pickle
import json
from flask import Flask, request

app = Flask(__name__)

@app.route("/predict")
def predict():
	input = request.args['input']
	input = int(input)
	filename = 'my_model.sav'
	model = pickle.load(open(filename, 'rb'))
	output = int(model.predict(input))
	return json.dumps(output)

if __name__ == '__main__':
	app.run(debug=True,port=80,host='0.0.0.0')
