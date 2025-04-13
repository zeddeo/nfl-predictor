from flask import Flask, request, jsonify
from preprocess import load_and_prepare
from predict import train_and_predict

app = Flask(__name__)

# Load and prep data once
df = load_and_prepare('nfl_gamelogs_over_under.csv')

@app.route('/')
def index():
    return "NFL Over/Under & Win Prediction API"

@app.route('/predict', methods=['GET'])
def predict():
    data = request.get_json()
    season = data.get('season')
    week = data.get('week')

    if not season or not week:
        return jsonify({'error': 'Missing season or week'}), 400

    under, win, acc_u, acc_w = train_and_predict(df, int(season), int(week))

    if under is None:
        return jsonify({'error': 'No test data for that week'}), 404

    return jsonify({
        'under_predictions': under,
        'win_predictions': win,
        'under_accuracy': acc_u,
        'win_accuracy': acc_w
    })

if __name__ == '__main__':
    app.run(debug=True)
