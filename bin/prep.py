# Third-party libraries
import pandas as pd


def convert_to_csf_or_pb(subset):
    if 'CSF' in subset:
        return 1
    else:
        return 0


column_name = ['vgene', 'jgene', 'dgene', 'cdr3', 'count', 'subset']
csf = pd.read_csv("../input/csf_cdr3.csv", header=None)
csf.columns = column_name
csf = csf.loc[:, ('cdr3', 'subset')]

print(csf.head())
pb = pd.read_csv("../input/pb_cdr3.csv", header=None)
pb.columns = column_name
pb = pb.loc[:, ('cdr3', 'subset')]
pb_sample = pb.sample(n=3500)
print(pb_sample.head())
csf_sample = csf.sample(n=3500)
concat = pd.concat([pb_sample, csf_sample])
concat['subset'] = concat['subset'].apply(convert_to_csf_or_pb)
print(concat.head())
concat.to_csv("../input/dataset.csv", index=False)
