# REQUIREMENTS
# $ pip install scipy
# $ pip install "polars[all]"
# You may need to install snappy in order to run this script:
# $ sudo pacman -S snappy
# $ pip install python-snappy

import polars as pl
import numpy as np
from scipy.io.wavfile import write

import os
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

if not os.path.isdir('wavs'):
	os.makedirs('wavs')

columns = ['ytid', 'ytid_seg', 'start', 'end', 'sentiment', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

df = pl.read_parquet('sqe_messai.parquet', columns = columns)

df.write_csv('sqe_messai_nowav.csv')

print(df)

columns2 = ['ytid_seg', 'wav2numpy']

df2 = pl.read_parquet('sqe_messai.parquet', use_pyarrow=False, columns = columns2)

def numpy2wav(row):
	segment = os.path.splitext(os.path.basename(os.path.normpath(row[0])))[0]
	print('PROCESSED:', segment)
	write('wavs/' + segment + '.wav', 16000, np.array(row[1]))
	return segment

df2.apply(lambda x: numpy2wav(x))
