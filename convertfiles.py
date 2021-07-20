import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input file")
ap.add_argument("-o", "--output", default='./', required=False,
                help="output directory, make sure to end with '/' ")
ap.add_argument("-f", "--function", default='PKL2CSV', required=False,
                help="what function to perform on file containing a pandas df, one of 'PKL2CSV', 'CSV2PKL', 'PKL2JSON', 'JSON2PKL'")
args = vars(ap.parse_args())

FILENAME = args['input']
LABEL = FILENAME.split('/')[-1].split('.')[0]
FUNCTION = args['function']
OUTDIR = args['output']
df = pd.read_pickle(FILENAME)

if FUNCTION == 'PKL2CSV':
    df = pd.read_pickle(FILENAME)
    df.to_csv(OUTDIR+LABEL+'.csv')
elif FUNCTION == 'CSV2PKL':
    df = pd.read_csv(FILENAME)
    df.to_pickle(OUTDIR+LABEL+'.pkl')
elif FUNCTION == 'PKL2JSON':
    df = pd.read_pickle(FILENAME)
    df.to_json(OUTDIR+LABEL+'.json')
elif FUNCTION == 'JSON2PKL':
    df = pd.read_json(FILENAME)
    df.to_pickle(OUTDIR+LABEL+'.pkl')
else:
    print("Function must be one of 'PKL2CSV', 'CSV2PKL', 'PKL2JSON', 'JSON2PKL'")
    exit
