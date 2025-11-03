from flask import Flask, request, jsonify
import sys
sys.path.append(".")
from src.experiments.tools.test import Inference
from src.utils.utils import get_config
import joblib
import numpy as np 


app = Flask(__name__)

inference = Inference(get_config())

@app.route("/enhance", methods=["POST"])
def enhance_signal():
    try:
        data = request.get_json()
        corrupted_signal = data.get("corrupted_signal", [])
        if not corrupted_signal:
            return jsonify({"error": "Invalid or missing 'corrupted_signal' field."}), 400

        reconstructed_signal = inference.infer(corrupted_signal)

        return jsonify({"reconstructed_signal": reconstructed_signal.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess_signal():
    try:
        data = request.get_json()
        signal = data.get('signal', [])
        signal = np.array(signal)
        return jsonify({'normalized_signal': signal.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/denormalize', methods=['POST'])
def denormalize_signal():
    try:
        normalized_signal = request.get_json().get('normalized_signal', []) 
        normalized_signal = np.array(normalized_signal)
        return jsonify({'denormalized_signal': normalized_signal.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
