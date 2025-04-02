from flask import Flask, render_template, request, jsonify
from model import MLBModel
from data_processor import DataProcessor
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize data processor and model
data_processor = DataProcessor()
model = MLBModel()

@app.route('/')
def index():
    # Get list of available batters and pitchers
    batters = data_processor.get_available_batters()
    pitchers = data_processor.get_available_pitchers()
    return render_template('index.html', batters=batters, pitchers=pitchers)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        batter = data.get('batter')
        pitcher = data.get('pitcher')
        
        if not batter or not pitcher:
            return jsonify({'error': 'Please select both a batter and pitcher'}), 400
        
        # Process the data and make prediction
        features = data_processor.prepare_features(batter, pitcher)
        prediction, confidence = model.predict(features)
        
        return jsonify({
            'prediction': float(prediction),
            'confidence': float(confidence),
            'batter': batter,
            'pitcher': pitcher
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 