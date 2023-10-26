
from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load your trained model
bst = xgb.Booster()  # Init model
bst.load_model('flaml_xgboost_model.model')  # Load trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    player_card_1 = int(data['player_card_1'])
    player_card_2 = int(data['player_card_2'])
    dealer_card_1 = int(data['dealer_card_1'])
    dealer_card_2 = int(data['dealer_card_2'])
    
    # Convert input data to numpy array and make prediction
    input_data = np.array([[player_card_1, player_card_2, dealer_card_1, dealer_card_2]])
    pred_probs = bst.predict(xgb.DMatrix(input_data))
    
    # Assuming the model predicts probabilities for class labels in the order [0, 1]
    # where 0 is "Hit" and 1 is "Stand"
    prob_stand = float(pred_probs[0])  # Convert float32 to standard float
    
    # Convert prediction probability to action
    # You can adjust the threshold if needed
    threshold = 0.7
    action = "Stand" if prob_stand > threshold else "Hit"
    
    return jsonify(action=action, probability=prob_stand)

if __name__ == '__main__':
    app.run(debug=True)

