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
# User all available cores
# It seems useless in the context of this script
n_cores = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores

# Create the wavs dir if it does not exist
if not os.path.isdir('wavs'):
	os.makedirs('wavs')

# All columns from the parquet file except the one with the audio numpy arrays (it is huge)
columns = ['ytid', 'ytid_seg', 'start', 'end', 'sentiment', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

### Add replace line here to change the paths from the ytid_seg column ###
# https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.str.replace.html
# replace the subtring '/home/emoman/Downloads/mosei/samples' with the actual path

# Read the parquet file with polars
df = pl.read_parquet('sqe_messai.parquet', columns = columns)

# Export the csv file (excluding the last column)
df.write_csv('sqe_messai_nowav.csv')
print(df)

# Now we are only interested on the column with the paths and the audio numpy arrays
columns2 = ['ytid_seg', 'wav2numpy']

# Read the parquet file with polars (this will take a while)
df2 = pl.read_parquet('sqe_messai.parquet', use_pyarrow=False, columns = columns2)

### Add replace line here to change the paths from the ytid_seg column ###
# https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.str.replace.html
# replace the subtring '/home/emoman/Downloads/mosei/samples' with the actual path

# Function to convert the numpy arrays to wav files stored in the wavs folders
def numpy2wav(row):
	segment = os.path.splitext(os.path.basename(os.path.normpath(row[0])))[0]
	print('PROCESSED:', segment)
	write('wavs/' + segment + '.wav', 16000, np.array(row[1]))
	return segment

# Apply the function (this will take a while)
df2.apply(lambda x: numpy2wav(x))
