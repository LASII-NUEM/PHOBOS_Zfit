import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your existing PHOBOS utils
from utils import ECM_utils, fitting_utils, linKK

app = Flask(__name__)
# Enable CORS so your GitHub Pages frontend can send requests
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "PHOBOS_Zfit API",
        "endpoints": ["/api/fit", "/api/simulate", "/api/linkk"]
    })

@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json(force=True)
        # Expected inputs:
        # frequencies: list of float
        # circuit: string (e.g. "R0-p(R1,C1)")
        # parameters: dict (e.g. {"R0": 10.0, "R1": 100.0, "C1": 1e-6})
        frequencies = np.array(data.get("frequencies", []), dtype=float)
        circuit = data.get("circuit", "")
        parameters = data.get("parameters", {})

        # Compute impedance using your ECM_utils
        # (adapt function names to match your ECM_utils definitions)
        z_complex = ECM_utils.calculate_impedance(circuit, parameters, frequencies)

        return jsonify({
            "success": True,
            "frequencies": frequencies.tolist(),
            "z_real": np.real(z_complex).tolist(),
            "z_imag": np.imag(z_complex).tolist(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/fit", methods=["POST"])
def fit():
    try:
        data = request.get_json(force=True)
        # Expected inputs:
        # frequencies: [f1, f2, ...]
        # z_real: [zr1, zr2, ...]
        # z_imag: [zi1, zi2, ...]
        # circuit: string
        # initial_params: dict
        # bounds: dict (optional)
        # method: string ('leastsq', 'nelder-mead', etc.)
        freq = np.array(data["frequencies"], dtype=float)
        z_real = np.array(data["z_real"], dtype=float)
        z_imag = np.array(data["z_imag"], dtype=float)
        z_exp = z_real + 1j * z_imag
        
        circuit = data.get("circuit")
        initial_params = data.get("initial_params", {})
        bounds = data.get("bounds", None)
        method = data.get("method", "leastsq")

        # Run your fitting routine from fitting_utils
        result = fitting_utils.fit_nyquist(
            freq=freq,
            z_exp=z_exp,
            circuit=circuit,
            initial_params=initial_params,
            bounds=bounds,
            method=method
        )

        return jsonify({
            "success": True,
            "fitted_params": result.get("params"),
            "chi_squared": result.get("chi_sq"),
            "residuals": result.get("residuals"),
            "fitted_z_real": np.real(result.get("z_fit")).tolist(),
            "fitted_z_imag": np.imag(result.get("z_fit")).tolist()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/linkk", methods=["POST"])
def linkk():
    try:
        data = request.get_json(force=True)
        freq = np.array(data["frequencies"], dtype=float)
        z_real = np.array(data["z_real"], dtype=float)
        z_imag = np.array(data["z_imag"], dtype=float)
        
        # Run your Lin-KK validation from linKK.py
        result = linKK.run_linKK(freq, z_real + 1j * z_imag)
        
        return jsonify({
            "success": True,
            "residuals_real": result.get("residuals_real").tolist(),
            "residuals_imag": result.get("residuals_imag").tolist(),
            "chi_squared": float(result.get("chi_squared", 0.0)),
            "is_valid": bool(result.get("is_valid", True))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
