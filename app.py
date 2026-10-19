from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the packaged model we created in Task 4
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the patient data sent to the API
        data = request.get_json()
        
        # 2. Convert it into the format the model expects
        df = pd.DataFrame([data])
        
        # 3. Make the prediction (0 = No Disease, 1 = Disease)
        prediction = model.predict(df)[0]
        
        # 4. Calculate the confidence (probability) of that prediction
        probabilities = model.predict_proba(df)[0]
        confidence = probabilities[prediction]
        
        # 5. Send the result back
        return jsonify({
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Run the API on port 5000
    app.run(host='0.0.0.0', port=5000)