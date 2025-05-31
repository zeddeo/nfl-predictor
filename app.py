from flask import Flask, request, jsonify, render_template
from preprocess import load_and_prepare
from predict import train_and_predict
import os

app = Flask(__name__)

# Load and prep data once
df = load_and_prepare('nfl_gamelogs_over_under.csv')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Looks for templates/index.html

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    season = data.get('Season')
    week = data.get('Week')

    if season is None or week is None:
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
    port = int(os.environ.get('PORT', 10000))  # Render uses this
    app.run(host='0.0.0.0', port=port, debug=True)
