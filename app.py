from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import urldl
import applcation

UPLOAD_FOLDER = "datapool"
ALLOWED_EXTENSIONS = set(['srt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return render_template('runoob.html')

@app.route('/wc/<filename>')
def wc(filename):
    return send_from_directory('./wc', filename, as_attachment=True)

@app.route('/upload/', methods=['POST'])
def upload():
    file = request.files['SRT']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    output=applcation.prosess(path, var = request.form.get('vars'), lenvar = request.form.get('lenvar'))
    return jsonify(message=output)

@app.route('/url/', methods=['POST'])
def url():
    insertValues = request.get_json()
    url = insertValues['url']
    urldl.dl(url)
    output=applcation.prosess('./datapool/tmp.srt', var=float(insertValues['vars']), lenvar=int(insertValues['lenvar']))
    return jsonify(message=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', 
            port=6006, 
            debug=False, 
            threaded=False
           )



