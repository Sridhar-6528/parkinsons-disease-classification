import os
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import predict, CLASS_NAMES

app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, "templates"))
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, "uploads")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PREDICT_DATA_DIR = os.path.join(PROJECT_ROOT, "predict_data")
MRI_DIR = os.path.join(PREDICT_DATA_DIR, "mri")
SPECT_DIR = os.path.join(PREDICT_DATA_DIR, "spect")


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample_files', methods=['GET'])
def get_sample_files():
    mri_files = []
    spect_files = []

    if os.path.exists(MRI_DIR):
        mri_files = sorted([f for f in os.listdir(MRI_DIR) if f.endswith(('.nii', '.nii.gz', '.pt', '.dcm', '.png', '.jpg', '.jpeg'))])
    if os.path.exists(SPECT_DIR):
        spect_files = sorted([f for f in os.listdir(SPECT_DIR) if f.endswith(('.nii', '.nii.gz', '.pt', '.dcm', '.png', '.jpg', '.jpeg'))])

    return jsonify({
        'mri_files': mri_files,
        'spect_files': spect_files
    })


@app.route('/api/predict', methods=['POST'])
def run_predict():
    try:
        import importlib
        import predict as predict_module
        importlib.reload(predict_module)
        predict_func = predict_module.predict

        model_type = request.form.get('model_type', 'proposed')
        ckpt_version = request.form.get('ckpt_version', 'fine_tuned')
        updrs = float(request.form.get('updrs', 0.0))
        moca = float(request.form.get('moca', 26.0))
        age = float(request.form.get('age', 65.0))
        gender = int(request.form.get('gender', 1))
        ground_truth = request.form.get('ground_truth', None)
        if ground_truth is not None and ground_truth != '' and ground_truth != 'none':
            ground_truth = int(ground_truth)
        else:
            ground_truth = None

        mri_path = None
        spect_path = None
        mri_filename = None
        spect_filename = None

        # Check for uploaded MRI file
        if 'mri_file' in request.files and request.files['mri_file'].filename != '':
            f = request.files['mri_file']
            mri_filename = secure_filename(f.filename)
            mri_path = os.path.join(app.config['UPLOAD_FOLDER'], mri_filename)
            f.save(mri_path)
        elif request.form.get('selected_mri'):
            selected = request.form.get('selected_mri')
            mri_path = os.path.join(MRI_DIR, selected)
            mri_filename = selected

        # Check for uploaded SPECT file
        if 'spect_file' in request.files and request.files['spect_file'].filename != '':
            f = request.files['spect_file']
            spect_filename = secure_filename(f.filename)
            spect_path = os.path.join(app.config['UPLOAD_FOLDER'], spect_filename)
            f.save(spect_path)
        elif request.form.get('selected_spect'):
            selected = request.form.get('selected_spect')
            spect_path = os.path.join(SPECT_DIR, selected)
            spect_filename = selected

        res = predict_func(
            mri_path=mri_path,
            spect_path=spect_path,
            updrs=updrs,
            moca=moca,
            age=age,
            gender=gender,
            model_type=model_type,
            ground_truth=ground_truth,
            mri_filename=mri_filename,
            spect_filename=spect_filename,
            ckpt_version=ckpt_version
        )

        return jsonify(res)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Starting Parkinson's AI Web Diagnostic Application on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
