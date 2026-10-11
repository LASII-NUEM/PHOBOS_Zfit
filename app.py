import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your PHOBOS / EIS utilities
from utils import ECM_utils, fitting_utils, linKK, file_utils

app = Flask(__name__)
# Enable CORS so your web frontend can send cross-origin requests
CORS(app)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "PHOBOS_Zfit API",
        "endpoints": ["/api/read_file", "/api/simulate", "/api/fit", "/api/linkk"]
    })


# ==============================================================================
# 1. FILE UPLOAD & PARSER ENDPOINT (EIS chemistry & PHOBOS electrical data)
# ==============================================================================
@app.route("/api/read_file", methods=["POST"])
@app.route("/api/read", methods=["POST"])
def read_file():
    """
    Receives an uploaded file via multipart/form-data and a 'type' string:
      - 'EIS': Chemistry data (reads frequency, Z_real, Z_imag)
      - 'PHOBOS': Electrical / dielectric data (reads or calculates Cp, Rp, freq, Z)
    """
    temp_path = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded in form-data"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename provided"}), 400

        # Extract data type: 'EIS' (default) or 'PHOBOS'
        data_type = request.form.get("type", "EIS").upper()
        if data_type not in ["EIS", "PHOBOS"]:
            data_type = "PHOBOS" if "ELEC" in data_type else "EIS"

        # Save to a temporary file preserving extension (.csv, .xlsx, .txt, .dat)
        filename = secure_filename(file.filename) or "sample.csv"
        _, ext = os.path.splitext(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Call file_utils.read with the specified type ('EIS' or 'PHOBOS')
        # Example: spec_obj = file_utils.read(temp_path, type="EIS")
        spec_obj = file_utils.read(temp_path, type=data_type)

        frequencies = None
        z_real = None
        z_imag = None
        cp = None
        rp = None

        # -------------------------------------------------------------
        # Robustly extract frequencies and impedance from spec_obj
        # -------------------------------------------------------------
        # 1. Frequencies
        for attr in ["freq", "frequency", "frequencies", "f"]:
            if hasattr(spec_obj, attr):
                frequencies = getattr(spec_obj, attr)
                break
            elif isinstance(spec_obj, dict) and attr in spec_obj:
                frequencies = spec_obj[attr]
                break

        # 2. Impedance components (Z_real, Z_imag or complex Z)
        for attr in ["Z_real", "z_real", "real", "Re"]:
            if hasattr(spec_obj, attr):
                z_real = getattr(spec_obj, attr)
                break
            elif isinstance(spec_obj, dict) and attr in spec_obj:
                z_real = spec_obj[attr]
                break

        for attr in ["Z_imag", "z_imag", "imag", "Im"]:
            if hasattr(spec_obj, attr):
                z_imag = getattr(spec_obj, attr)
                break
            elif isinstance(spec_obj, dict) and attr in spec_obj:
                z_imag = spec_obj[attr]
                break

        # Check for complex impedance 'Z'
        if (z_real is None or z_imag is None):
            for attr in ["Z", "z", "impedance"]:
                if hasattr(spec_obj, attr):
                    z_complex = getattr(spec_obj, attr)
                    z_real = np.real(z_complex)
                    z_imag = np.imag(z_complex)
                    break
                elif isinstance(spec_obj, dict) and attr in spec_obj:
                    z_complex = spec_obj[attr]
                    z_real = np.real(z_complex)
                    z_imag = np.imag(z_complex)
                    break

        # -------------------------------------------------------------
        # PHOBOS Electrical features: Cp, Rp, freq, and impedance
        # -------------------------------------------------------------
        if data_type == "PHOBOS" or hasattr(spec_obj, "Cp") or (isinstance(spec_obj, dict) and "Cp" in spec_obj):
            for attr in ["Cp", "cp", "capacitance"]:
                if hasattr(spec_obj, attr):
                    cp = getattr(spec_obj, attr)
                    break
                elif isinstance(spec_obj, dict) and attr in spec_obj:
                    cp = spec_obj[attr]
                    break

            for attr in ["Rp", "rp", "resistance"]:
                if hasattr(spec_obj, attr):
                    rp = getattr(spec_obj, attr)
                    break
                elif isinstance(spec_obj, dict) and attr in spec_obj:
                    rp = spec_obj[attr]
                    break

            # If Cp and Rp were provided without Z, calculate impedance:
            # Y = 1/Rp + j*w*Cp => Z = 1/Y
            if (z_real is None or z_imag is None) and (cp is not None and rp is not None and frequencies is not None):
                f_arr = np.array(frequencies, dtype=float)
                cp_arr = np.array(cp, dtype=float)
                rp_arr = np.array(rp, dtype=float)
                omega = 2.0 * np.pi * f_arr
                y_complex = (1.0 / rp_arr) + 1j * (omega * cp_arr)
                z_complex = 1.0 / y_complex
                z_real = np.real(z_complex)
                z_imag = np.imag(z_complex)

            # Conversely, if Z is available but Cp/Rp are missing, calculate them
            elif (cp is None or rp is None) and (z_real is not None and z_imag is not None and frequencies is not None):
                f_arr = np.array(frequencies, dtype=float)
                zr_arr = np.array(z_real, dtype=float)
                zi_arr = np.array(z_imag, dtype=float)
                omega = 2.0 * np.pi * f_arr
                denom = zr_arr**2 + zi_arr**2
                with np.errstate(divide='ignore', invalid='ignore'):
                    g = np.where(denom > 0, zr_arr / denom, 0)
                    b = np.where(denom > 0, -zi_arr / denom, 0)
                    rp = np.where(g != 0, 1.0 / g, np.nan)
                    cp = np.where(omega > 0, b / omega, np.nan)

        if frequencies is None or z_real is None or z_imag is None:
            return jsonify({
                "success": False,
                "error": f'Could not extract frequency and impedance components from "{filename}". Check file format.'
            }), 400

        # Convert all NumPy arrays to Python native lists
        freq_list = np.asarray(frequencies, dtype=float).tolist()
        zr_list = np.asarray(z_real, dtype=float).tolist()
        zi_list = np.asarray(z_imag, dtype=float).tolist()
        cp_list = np.asarray(cp, dtype=float).tolist() if cp is not None else None
        rp_list = np.asarray(rp, dtype=float).tolist() if rp is not None else None

        return jsonify({
            "success": True,
            "type": data_type,
            "filename": filename,
            "frequencies": freq_list,
            "z_real": zr_list,
            "z_imag": zi_list,
            "cp": cp_list,
            "rp": rp_list,
            "data": {
                "name": os.path.splitext(filename)[0],
                "originalFilename": filename,
                "points": [
                    {
                        "freq": float(f),
                        "omega": float(2.0 * np.pi * f),
                        "zReal": float(zr),
                        "zImag": float(zi),
                        "zImagNeg": float(-zi),
                        "zMag": float(np.sqrt(zr**2 + zi**2)),
                        "phaseDeg": float(np.degrees(np.arctan2(zi, zr))),
                        "cp": float(cp_list[i]) if cp_list and not np.isnan(cp_list[i]) else None,
                        "rp": float(rp_list[i]) if rp_list and not np.isnan(rp_list[i]) else None
                    }
                    for i, (f, zr, zi) in enumerate(zip(freq_list, zr_list, zi_list))
                ]
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
    finally:
        # Clean up temporary disk file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ==============================================================================
# 2. LINEAR KRAMERS-KRONIG (LIN-KK) VALIDATION ENDPOINT
# ==============================================================================
@app.route("/api/linkk", methods=["POST"])
def linkk_validation():
    try:
        data = request.get_json(force=True)
        freq = np.array(data.get("frequencies", []), dtype=float)
        z_real = np.array(data.get("z_real", []), dtype=float)
        z_imag = np.array(data.get("z_imag", []), dtype=float)
        c = float(data.get("c", 0.1))
        max_iter = int(data.get("max_iterations", data.get("max_iter", 50)))
        add_capacitor = bool(data.get("add_capacitor", data.get("addCapacitor", True)))

        if len(freq) < 3:
            return jsonify({"success": False, "error": "At least 3 frequency points are required"}), 400

        z_exp = z_real + 1j * z_imag

        # Invoke linKK from your utils
        result = None
        if hasattr(linKK, "run_linKK"):
            try:
                result = linKK.run_linKK(freq, z_exp, c=c, max_iter=max_iter, add_capacitor=add_capacitor)
            except TypeError:
                result = linKK.run_linKK(freq, z_exp)
        elif hasattr(linKK, "LinKK"):
            result = linKK.LinKK(freq, z_real, z_imag)
        else:
            raise AttributeError("linKK module has no recognized run_linKK or LinKK function")

        # Extract real & imaginary residuals
        res_real = result.get("residuals_real") if isinstance(result, dict) else getattr(result, "residuals_real", None)
        res_imag = result.get("residuals_imag") if isinstance(result, dict) else getattr(result, "residuals_imag", None)
        chi_sq = float(result.get("chi_squared", result.get("pseudo_chi_squared", 0.0)) if isinstance(result, dict) else getattr(result, "chi_squared", 0.0))
        is_valid = bool(result.get("is_valid", chi_sq < 1e-4) if isinstance(result, dict) else getattr(result, "is_valid", chi_sq < 1e-4))

        res_real_list = np.asarray(res_real, dtype=float).tolist() if res_real is not None else []
        res_imag_list = np.asarray(res_imag, dtype=float).tolist() if res_imag is not None else []

        max_res_real = float(np.max(np.abs(res_real_list))) if len(res_real_list) > 0 else 0.0
        max_res_imag = float(np.max(np.abs(res_imag_list))) if len(res_imag_list) > 0 else 0.0

        status = "Consistent" if is_valid else ("Marginal" if (max_res_real < 0.05 and max_res_imag < 0.05) else "Inconsistent")

        return jsonify({
            "success": True,
            "residuals_real": res_real_list,
            "residuals_imag": res_imag_list,
            "chi_squared": chi_sq,
            "is_valid": is_valid,
            "data": {
                "status": status,
                "freqs": freq.tolist(),
                "residualsReal": res_real_list,
                "residualsImag": res_imag_list,
                "pseudoChiSquare": chi_sq,
                "chiSquare": chi_sq,
                "maxResidualReal": max_res_real,
                "maxResidualImag": max_res_imag,
                "summary": f"Residuals max real: {max_res_real * 100:.2f}%, max imag: {max_res_imag * 100:.2f}%"
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==============================================================================
# 3. EQUIVALENT CIRCUIT FITTING ENDPOINT
# ==============================================================================
@app.route("/api/fit", methods=["POST"])
def fit_circuit():
    try:
        data = request.get_json(force=True)
        freq = np.array(data["frequencies"], dtype=float)
        z_real = np.array(data["z_real"], dtype=float)
        z_imag = np.array(data["z_imag"], dtype=float)
        z_exp = z_real + 1j * z_imag

        circuit = data.get("circuit") or data.get("expression", "")
        method = data.get("method") or data.get("algorithm", "leastsq")
        bounds = data.get("bounds", None)

        # Parse initial_params
        initial_params = data.get("initial_params", {})
        if not initial_params and "elements" in data:
            initial_params = {
                el["label"]: float(el["value"])
                for el in data["elements"]
                if "label" in el and "value" in el
            }

        # Run fitting routine
        if hasattr(fitting_utils, "fit_nyquist"):
            result = fitting_utils.fit_nyquist(
                freq=freq,
                z_exp=z_exp,
                circuit=circuit,
                initial_params=initial_params,
                bounds=bounds,
                method=method
            )
        elif hasattr(fitting_utils, "fit"):
            result = fitting_utils.fit(circuit, initial_params, freq, z_exp, method=method)
        else:
            raise AttributeError("fitting_utils module has no fit_nyquist or fit function")

        params = result.get("params", result.get("fitted_params", {})) if isinstance(result, dict) else {}
        chi_sq = float(result.get("chi_sq", result.get("chi_squared", 0.0)) if isinstance(result, dict) else 0.0)
        residuals = result.get("residuals", None) if isinstance(result, dict) else None

        z_fit = result.get("z_fit", None) if isinstance(result, dict) else None
        if z_fit is not None:
            fitted_zr = np.real(z_fit).tolist()
            fitted_zi = np.imag(z_fit).tolist()
        else:
            fitted_zr = result.get("fitted_z_real", []) if isinstance(result, dict) else []
            fitted_zi = result.get("fitted_z_imag", []) if isinstance(result, dict) else []

        return jsonify({
            "success": True,
            "fitted_params": params,
            "chi_squared": chi_sq,
            "residuals": residuals.tolist() if isinstance(residuals, np.ndarray) else residuals,
            "fitted_z_real": fitted_zr,
            "fitted_z_imag": fitted_zi,
            "data": {
                "algorithm": method,
                "circuitExpression": circuit,
                "converged": True,
                "iterations": result.get("iterations", 1) if isinstance(result, dict) else 1,
                "finalLoss": chi_sq,
                "chiSquare": chi_sq,
                "parameters": params,
                "fittedPoints": [
                    {
                        "freq": float(f),
                        "omega": float(2.0 * np.pi * f),
                        "zReal": float(zr),
                        "zImag": float(zi),
                        "zImagNeg": float(-zi),
                        "zMag": float(np.sqrt(zr**2 + zi**2)),
                        "phaseDeg": float(np.degrees(np.arctan2(zi, zr)))
                    }
                    for f, zr, zi in zip(freq, fitted_zr, fitted_zi)
                ]
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==============================================================================
# 4. EQUIVALENT CIRCUIT SIMULATION ENDPOINT
# ==============================================================================
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
