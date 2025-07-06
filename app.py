# app.py (Flask backend)
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
from gnss_utils import process_all

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Home route (landing page)
@app.route('/')
def home():
    return render_template('index.html')

# 2. Upload page (form)
@app.route('/upload')  # GET method only
def show_upload_page():
    return render_template('upload.html')

# 3. File upload handling
@app.route('/upload', methods=['POST'])
def upload_files():
    gps_file = request.files.get('gpsFile')
    navic_file = request.files.get('navicFile')

    if not gps_file or not navic_file:
        return jsonify({"error": "Both GPS and NAVIC files are required."}), 400

    gps_path = os.path.join(UPLOAD_FOLDER, "gps.csv")
    navic_path = os.path.join(UPLOAD_FOLDER, "navic.csv")

    gps_file.save(gps_path)
    navic_file.save(navic_path)

    # ✅ Redirect to result display page
    return redirect(url_for("position"))

# 4. Result route
@app.route('/position')
def position():
    result = process_all(
        gps_csv=os.path.join(UPLOAD_FOLDER, 'gps.csv'),
        navic_csv=os.path.join(UPLOAD_FOLDER, 'navic.csv'),
        true_coords=[225003486.6, 34911968.68, 6711011.808]
    )

    if result is None:
        return jsonify({"error": "Model failed or no LOS satellites"}), 500

    # ✅ Only ONE return allowed: either render_template or jsonify
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
