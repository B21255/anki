from _datetime import date, datetime
import json
import xlsxwriter
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
from io import StringIO


# Paths to folders
upload_filepath_csv = os.path.join(
    'custom_directory', 'uploaded_files', 'csv_files')
upload_filepath_fasta = os.path.join(
    'custom_directory', 'uploaded_files', 'fasta_files')
model_filepath = os.path.join('custom_directory', 'saved_models')
model_results = os.path.join('custom_directory', 'results')

# extension allowed
ALLOWED_EXTENSIONS = {'fasta'}

# Flask App
app = Flask(__name__)

app.config['upload_filepath_csv'] = upload_filepath_csv
app.config['upload_filepath_fasta'] = upload_filepath_fasta
app.config['model_filepath'] = model_filepath
app.config['model_results'] = model_results

# functions


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# define different functions for sequence to numeric representation
nucleic_acid_list = ['A', 'T', 'G', 'C']
nucleic_acid_dict = dict()
for s in nucleic_acid_list:
    nucleic_acid_dict[s] = nucleic_acid_list.index(s) + 1

amino_acid_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                   'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_acid_dict = dict()
for fasta_sequences in amino_acid_list:
    amino_acid_dict[fasta_sequences] = amino_acid_list.index(
        fasta_sequences) + 1

# defining fasta to dataframe


def fasta_to_dataframe(upload_filepath_fasta=None):
    L = list()
    # L.append(['seq_id', 'seq', 'seq_len'])
    with open(upload_filepath_fasta) as handle:
        for record in SeqIO.parse(handle=handle, format='fasta'):
            seq = ''.join(record.seq)
            L.append([record.id, seq, len(seq)])
    df = pd.DataFrame(L, columns=['seq_id', 'seq', 'seq_len'])
    df = df.drop_duplicates(subset=['seq'], keep='first')
    return df
# defining window size


def get_fixed_width_seq(window_size=True, seq_df=None):
    # def get_fixed_width_seq(window_size=32, seq_df=None):
    List = list()
    for i, seq in enumerate(seq_df.seq):
        if len(seq) <= window_size:
            List.append([seq_df.seq_id.values[i], seq, len(seq)])
            continue
        seq_id = seq_df.seq_id.values[i]
        for j in range(len(seq)-window_size):
            List.append([seq_id+'_duplicate_'+str(j),
                        seq[j:j+window_size], window_size])
    df1 = pd.DataFrame(List, columns=seq_df.columns)
    df1 = df1.drop_duplicates(subset=['seq'], keep='first')
    df1 = df1.reset_index(drop=True)
    return df1
# define amino acid seq


def uniq_aa(seq_df=None):
    uniq_symb = set()
    for seq in seq_df.seq:
        uniq_symb = uniq_symb.union(list(seq))
    return sorted(list(uniq_symb))
# define droping of invalid seq


def drop_invalid_seq_1(sub_df=None, ref_df=None):
    cols = sub_df.columns
    sub_df['invalid_seq'] = [1 if seq in ref_df['seq']
                             else 0 for seq in sub_df['seq']]
    return sub_df.loc[sub_df['invalid_seq'] == 0, cols].reset_index(drop=True)
# define droping of invalid seq


def drop_invalid_seq_2(sub_df=None, aa_list=None):
    cols = sub_df.columns
    sub_df['invalid_seq'] = [1 if set(list(seq)).union(
        aa_list) != set(aa_list) else 0 for seq in sub_df['seq']]
    return sub_df.loc[sub_df['invalid_seq'] == 0, cols].reset_index(drop=True)

# amin acid seq to encoding


def seq_2_array(df, aa_list=amino_acid_list):
    max_len = df.seq_len.max()
    # print(df.seq)
    # print(aa_list)
    L = list()
    i = 0

    for seq in df.seq:
        # L.append(df.seq_id[i])
        if len(seq) < max_len:
            L.append([df.seq_id[i]]+[df.seq[i]]+[aa_list.index(l) +
                     1 for l in list(seq)]+[0]*(max_len-len(seq)))
            i = i + 1
        else:
            L.append([df.seq_id[i]]+[df.seq[i]] +
                     [aa_list.index(l)+1 for l in list(seq)])
            i = i + 1

    return np.array(L)

# Dataset Prepocessing


def data_preprocess(f):
    filename = f
    data = pd.read_csv(os.path.join(
        app.config['upload_filepath_csv'], filename), header=None)
    seq_id = data.iloc[:, 0].values
    seq_s = data.iloc[:, 1].values
    x = data.iloc[:, 2:].values  # all rows and 2nd to all remaining coloumn

    preprocess = StandardScaler()
    x = preprocess.fit_transform(x)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, seq_id, seq_s

# function that extract features from fasta file to a csv file


def fastaToCSV(f):
    fasta_file = f
    f, e = os.path.splitext(fasta_file)
    fasta_file_path = os.path.join(
        app.config['upload_filepath_fasta'], fasta_file)
    seq = fasta_to_dataframe(str(fasta_file_path))
    data = get_fixed_width_seq(window_size=25, seq_df=seq)
    # print(data)
    valp = seq_2_array(data, amino_acid_list)
    # print(valp)
    csv_file = f+'.csv'
    with open(os.path.join(app.config['upload_filepath_csv'], csv_file), 'w') as f11:
        writer = csv.writer(f11)
        writer.writerows(valp)
    return csv_file

# Home page


@app.route('/', methods=['POST', 'GET'])
def home_screen():
    return render_template('test.html')

# Dataset upload


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if request.form and request.files is None:
            return json.dumps({
                'status': 'NOT OK'
            })
        elif 'file' in request.files:
            datafile = request.files['file']
            if (allowed_file(datafile.filename)):
                if datafile:
                    filename = secure_filename(datafile.filename)
                    f, e = os.path.splitext(filename)
                    today = datetime.now()
                    today = today.strftime("%d_%m_%y_%H_%M_%S")
                    newFileNme = f + '_' + today + e
                    datafile.save(os.path.join(
                        app.config['upload_filepath_fasta'], newFileNme))
                    new_csv_file = fastaToCSV(newFileNme)
                    return json.dumps({
                        'status': 'OK',
                        'file_name': new_csv_file
                    })
                else:
                    return json.dumps({
                        'status': 'NOT OK'
                    })
            else:
                return json.dumps({
                    'status': 'NOT CSV'
                })
        elif 'sequence' in request.form:
            seq = request.form['sequence']
            seq_str = StringIO(seq)
            today = datetime.now()
            today = today.strftime("%d_%m_%y_%H_%M_%S")
            seq_filename = 'pasted_seq_' + today + '.fasta'

            file = open(os.path.join(
                app.config['upload_filepath_fasta'], seq_filename), "w+")
            file.write(seq)
            file.close
            new_csv_file = fastaToCSV(seq_filename)
            return json.dumps({
                'status': 'OK',
                'file_name': new_csv_file
            })
        else:

            return json.dumps({
                'status': 'NOT OK'
            })
    else:
        return json.dumps({
            'status': 'NOT OK'
        })

# Prediction with the saved models


@app.route('/model', methods=['POST', 'GET'])
def loading_model():
    filename = None
    if request.method == 'POST':
        if request.form is None:
            return json.dumps({
                'status': 'NOT OK'
            })
        else:
            filename = request.form['file_name']
            f, e = os.path.splitext(filename)
            x, seq_id, seq = data_preprocess(filename)
            final_result = pd.DataFrame(seq_id, columns=['Sequence_ID'])
            final_result['sequences'] = seq

            my_model = load_model(os.path.join(
                app.config['model_filepath'], 'model.h5'))
            y_pred = my_model.predict(x)
            new_y = []
            for val in y_pred:
                if val >= 0.5:
                    new_y.append('Present')
                else:
                    new_y.append('Absent')
            final_result['preds'] = new_y
            data = final_result.to_json(orient='records')
            return data
    else:
        return json.dumps({
            'status': 'NOT OK'
        })


# run the api
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=False)
