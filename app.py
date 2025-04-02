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
    # Get list of all teams
    teams = data_processor.get_all_teams()
    
    # Create a dict structure for the frontend with teams and their players
    team_data = {}
    for team in teams:
        batters = data_processor.get_team_batters(team)
        pitchers = data_processor.get_team_pitchers(team)
        if batters or pitchers:  # Only include teams that have players
            team_data[team] = {
                'batters': batters,
                'pitchers': pitchers
            }
    
    return render_template('index.html', team_data=team_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        batter_team = data.get('batter_team')
        batter_name = data.get('batter_name')
        pitcher_team = data.get('pitcher_team')
        pitcher_name = data.get('pitcher_name')
        
        if not batter_team or not batter_name or not pitcher_team or not pitcher_name:
            return jsonify({'error': 'Please select both a batter and pitcher'}), 400
        
        # Process the data and make prediction
        features = data_processor.prepare_matchup_features(
            batter_team, batter_name, pitcher_team, pitcher_name
        )
        prediction, confidence = model.predict_matchup(features)
        
        return jsonify({
            'prediction': float(prediction),
            'confidence': float(confidence),
            'batter': batter_name,
            'pitcher': pitcher_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 