from flask import Flask, request
import joblib
import numpy
import sklearn

MODEL_PATH = 'ml/model.pkl'
SCALER_X_PATH = 'ml/scaler_x.pkl'
SCALER_Y_PATH = 'ml/scaler_y.pkl'

app = Flask(__name__)

@app.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args
    area = args.get('area', default=-1, type=float)
    studio = args.get('studio', default=-1, type=int)
    renovation = args.get('renovation', default=-1, type=int)
    day_difference_int = args.get('day_difference_int', default=-1, type=int)

    #response = "area:{}, studio:{}, renovation:{}, day_difference_int:{}".format(area, studio, renovation, day_difference_int)

    model = joblib.load(MODEL_PATH)
    sc_x = joblib.load(SCALER_X_PATH)
    sc_y = joblib.load(SCALER_Y_PATH)

    x = numpy.array([area, studio, renovation, day_difference_int]).reshape(1,-1)
    x = sc_x.transform(x)

    result = model.predict(x)
    result = numpy.exp(sc_y.inverse_transform(result.reshape(1,-1)))

    return str(result[0][0])

if __name__ == '__main__':
    app.run(debug = True, port = 5443, host = '0.0.0.0')