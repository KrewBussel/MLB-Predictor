# MLB Performance Predictor

A machine learning application that predicts the number of hits a selected MLB batter will achieve in their next game against a specific pitcher.

## Features

- Predicts hits for batter-pitcher matchups
- Uses historical statistical data
- Interactive web interface
- Real-time predictions
- Confidence level indicators

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your batter and pitcher CSV files in the `data/batters` and `data/pitchers` directories
4. Run the application:
   ```bash
   python app.py
   ```

## Project Structure

- `app.py`: Main Flask application
- `model.py`: Machine learning model implementation
- `data_processor.py`: Data processing and feature engineering
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and other static files
- `data/`: Directory for CSV files
  - `batters/`: Batter statistics CSV files
  - `pitchers/`: Pitcher statistics CSV files

## Data Format

### Batter Stats CSV
Headers: Rk, Gcar, Gtm, Date, Team, Opp, Result, Inngs, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GIDP, HBP, SH, SF, IBB, aLI, WPA, acLI, cWPA, RE24, DFS(DK), DFS(FD), BOP, Pos

### Pitcher Stats CSV
Headers: Rk, Gcar, Gtm, Date, Tm, Opp, Rslt, Inngs, Dec, DR, IP, H, R, ER, BB, SO, HR, HBP, ERA, FIP, BF, Pit, Str, StL, StS, GB, FB, LD, PU, Unk, GSc, IR, IS, SB, CS, PO, AB, 2B, 3B, IBB, GDP, SF, ROE, aLI, WPA, acLI, cWPA, RE24, DFS(DK), DFS(FD), Entered, Exited

## License

MIT License 