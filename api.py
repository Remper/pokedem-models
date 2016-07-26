import argparse
import numpy as np

from flask import Flask, request, json, g
from models.simple import SimpleModel

app = Flask(__name__)
model = None

@app.route("/predict")
def predict():
    try:
        features = request.args['features']
        if features is None:
            raise ValueError('provide a list of features')
        features = np.array(json.loads(features))
        result = model.predict(features=features)
        return json.jsonify(result.reshape([-1]).tolist())
    except Exception as e:
        print(e)
        return json.jsonify({
            'result': 'error',
            'message': e.message
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple API that returns predictions from a model')
    parser.add_argument('--input', default='model', help='Input model', metavar='#')
    args = parser.parse_args()

    print("Initialized with settings:")
    print(vars(args))

    print("Loading model")
    model = SimpleModel.restore_definition(args.input)
    model.batch_size(1)
    model.restore_from_file(args.input)

    print("Starting webserver")
    app.run()
