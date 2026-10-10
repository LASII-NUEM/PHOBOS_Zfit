import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your PHOBOS utils (including file_utils)
from utils import ECM_utils, fitting_utils, linKK, file_utils

app = Flask(__name__)
# Enable CORS so your GitHub Pages frontend can send requests
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "PHOBOS_Zfit API",
        "endpoints": ["/api/fit", "/api/simulate", "/api/linkk", "/api/read_file"]
    })

@app.route("/api/read_file", methods=["POST"])
@app.route("/api/read", methods=["POST"])
def read_file():
    """
    Parses an uploaded EIS or PHOBOS file using file_utils.read(filepath, type=file_type).
    Expects multipart/form-data:
      - 'file': the uploaded .csv or .xlsx file
      - 'type': 'EIS' (default) or 'PHOBOS'
    Returns JSON:
      - 'success': True
      - 'frequencies': [f1, f2, ...]
      - 'z_real': [zr1, zr2, ...]
      - 'z_imag': [zi1, zi2, ...]
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded in 'file' field"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename provided"}), 400

    file_type = request.form.get("type", "EIS")
    filename = secure_filename(uploaded_file.filename) or "data.csv"
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".csv"

    # Save to a temporary file so file_utils.read can open it from disk
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name

    try:
        uploaded_file.save(temp_path)
        temp_file.close()

        # Call PHOBOS file_utils.read
        spec_obj = file_utils.read(temp_path, type=file_type)

        freq = None
        z_real = None
        z_imag = None

        # 1. Flexible attribute extraction from spec_obj (SpectroscopyData / EISData)
        for f_attr in ["freq", "frequencies", "frequency", "f"]:
            if hasattr(spec_obj, f_attr):
                val = getattr(spec_obj, f_attr)
                if val is not None:
                    freq = np.asarray(val, dtype=float)
                    break

        for zr_attr in ["Z_real", "z_real", "real", "Re", "re"]:
            if hasattr(spec_obj, zr_attr):
                val = getattr(spec_obj, zr_attr)
                if val is not None:
                    z_real = np.asarray(val, dtype=float)
                    break

        for zi_attr in ["Z_imag", "z_imag", "imag", "Im", "im"]:
            if hasattr(spec_obj, zi_attr):
                val = getattr(spec_obj, zi_attr)
                if val is not None:
                    z_imag = np.asarray(val, dtype=float)
                    break

        # 2. Check for complex Z attribute if real/imag were not separate
        if hasattr(spec_obj, "Z") and (z_real is None or z_imag is None):
            z_complex = getattr(spec_obj, "Z")
            if z_complex is not None:
                z_arr = np.asarray(z_complex, dtype=complex)
                z_real = np.real(z_arr)
                z_imag = np.imag(z_arr)

        # 3. Handle dictionary returns if spec_obj is a dict
        if isinstance(spec_obj, dict):
            freq = spec_obj.get("frequencies") or spec_obj.get("freq") or spec_obj.get("f")
            z_real = spec_obj.get("z_real") or spec_obj.get("Z_real") or spec_obj.get("real")
            z_imag = spec_obj.get("z_imag") or spec_obj.get("Z_imag") or spec_obj.get("imag")
            if (z_real is None or z_imag is None) and "Z" in spec_obj:
                z_arr = np.asarray(spec_obj["Z"], dtype=complex)
                z_real = np.real(z_arr)
                z_imag = np.imag(z_arr)

        # 4. Check for 2D data matrix or DataFrame fallback
        if (freq is None or z_real is None or z_imag is None) and hasattr(spec_obj, "data"):
            data_matrix = spec_obj.data
            if hasattr(data_matrix, "values"):  # pandas DataFrame
                data_matrix = data_matrix.values
            if isinstance(data_matrix, np.ndarray) and data_matrix.shape[1] >= 3:
                freq = data_matrix[:, 0].astype(float)
                # If Autolab layout (col 0=f, col 5=Z', col 6=-Z'')
                if data_matrix.shape[1] >= 7:
                    z_real = data_matrix[:, 5].astype(float)
                    z_imag = -data_matrix[:, 6].astype(float)
                else:
                    z_real = data_matrix[:, 1].astype(float)
                    z_imag = data_matrix[:, 2].astype(float)

        if freq is None or z_real is None or z_imag is None:
            return jsonify({
                "success": False,
                "error": "Failed to extract frequency, Z_real, or Z_imag from file_utils.read output"
            }), 400

        # Convert all NumPy arrays to standard Python lists
        freq = np.asarray(freq, dtype=float).flatten()
        z_real = np.asarray(z_real, dtype=float).flatten()
        z_imag = np.asarray(z_imag, dtype=float).flatten()

        return jsonify({
            "success": True,
            "filename": uploaded_file.filename,
            "frequencies": freq.tolist(),
            "z_real": z_real.tolist(),
            "z_imag": z_imag.tolist()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
    finally:
        # Guarantee temporary file is deleted from disk
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json(force=True)
        frequencies = np.array(data.get("frequencies", []), dtype=float)
        circuit = data.get("circuit", "")
        parameters = data.get("parameters", {})

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
        freq = np.array(data["frequencies"], dtype=float)
        z_real = np.array(data["z_real"], dtype=float)
        z_imag = np.array(data["z_imag"], dtype=float)
        z_exp = z_real + 1j * z_imag
        
        circuit = data.get("circuit")
        initial_params = data.get("initial_params", {})
        bounds = data.get("bounds", None)
        method = data.get("method", "leastsq")

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
