# -*- coding: utf-8 -*-
# +
# # %pip install Flask
# # %pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# # %pip install transformers
# # %pip install bert4keras
# # %pip install pandas
# # %pip install sentencepiece
# # %pip install ipywidgets
# # %pip install sklearn
# # %pip install sentence_transformers
# # %pip install -U pip setuptools wheel
# # %pip install -U spacy
# # %pip install bert-extractive-summarizer
# # %pip install pytube
# # %pip install beautifulsoup4
# # %pip install wordcloud


# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("wordnet")
# nltk.download('averaged_perceptron_tagger')

# # 去terminal跑
# # !python -m spacy download en_core_web_trf

# +
from flask import Flask, render_template, jsonify, request, send_from_directory

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
# global graph
# import tensorflow
# graph = tensorflow.compat.v1.get_default_graph()

from werkzeug.utils import secure_filename
import os
import urldl
import applcation

# +
# os.environ["TF_KERAS"]="1"
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
#     print(file)
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    output=applcation.prosess(path, var = request.form.get('vars'), lenvar = request.form.get('lenvar'))
#     output=[['data!','QEDFCQEF QEF'], ['data2']]
    return jsonify(message=output)

@app.route('/url/', methods=['POST'])
def url():
    insertValues = request.get_json()
    url = insertValues['url']
#     print(type(insertValues['vars']), type(insertValues['lenvar']))
    urldl.dl(url)
    output=applcation.prosess('./datapool/tmp.srt', var=float(insertValues['vars']), lenvar=int(insertValues['lenvar']))
#     output=[['data!','QEDFCQEF QEF'], ['data2']]
    return jsonify(message=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', 
            port=6006, 
            debug=False, 
            threaded=False
           )

# +
# path = u"./dataset/s2t/EN_0304_1.srt"
# print(applcation.prosess(path))
# +
# import uuid
# my_uuid = str(uuid.uuid4())


# +
# my_uuid
# -


