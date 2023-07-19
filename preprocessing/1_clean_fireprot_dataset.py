import pandas as pd
import numpy as np


def drop_duplicate_entries(tmp):
    """Drop cases where mutation and ddG/dTm value is the same, but other info is different"""
    arr = np.empty(tmp.shape[0], dtype=object)

    # get unique entries by ddG/dTm value and other key info
    for n, i in enumerate(tmp.index):
        arr[n] = str(tmp['pdb_id_corrected'][i]) + '-' + str(tmp['mutation'][i]) + \
                 '-' + str(tmp['position'][i]) + '-' + str(tmp['wild_type'][i]) + '-' \
                 + str(tmp['uniprot_id'][i]) + '-' + str(tmp['pH'][i]) + '-' + str(tmp['ddG'][i]) + str(tmp['dTm'][i])
    tmp['dupe_detector'] = arr
    # drop duplicate ddG/dTm values for same mutation(s) and conditions
    tmp = tmp.drop_duplicates(subset=['dupe_detector'])
    return tmp


def reconcile_multi_pdbs(df, correction_file):
    """Reconcile cases w/multiple pdbs based on manual corrections file"""
    # remove duplicate codes for clarity
    pdb_ids = df['pdb_id']
    codes = [list(set(p.split('|'))) for p in pdb_ids]

    # identify multiple ID cases
    codes = np.array(codes, dtype=object)
    u, c = np.unique(codes, return_counts=True)
    df['pdb_id_corrected'] = ['|'.join(list(set(str(p).split('|')))) for p in df['pdb_id']]

    # load corrections
    corr_df = pd.read_csv(correction_file, header=0, encoding='unicode_escape', engine='python')

    for c in corr_df['pdb_id']:
        if df['pdb_id_corrected'][df['pdb_id'] == c].size < 1:
            break
        rv = corr_df['pdb_disambiguated'][corr_df['pdb_id'] == c]
        rv = rv.values[0]
        print('Replacing %s with %s' % (c, rv))
        df.loc[df['pdb_id'] == c, 'pdb_id_corrected'] = rv
    return df


def clean_fireprot_dataset(in_file, out_file, correction_file):
    # downloaded raw data from https://loschmidt.chemi.muni.cz/fireprotdb/
    df = pd.read_csv(in_file, header=0,
                     dtype={'is_curated': 'str',
                            'is_essential': 'str',
                            'is_back_to_consensus': 'str',
                            'is_in_catalytic_pocket': 'str',
                            'is_in_tunnel_bottleneck': 'str',
                            'method': 'str',
                            'method_details': 'str',
                            'technique': 'str',
                            'technique_details': 'str',
                            'notes': 'str'
                            })
    print('Raw dataset size:', df.shape)

    # drop duplicates
    df = df.drop_duplicates()
    print('After dropping duplicates:', df.shape)

    # drop rows with no ddG values
    # df = df.dropna(subset=['ddG'])
    df = df.dropna(subset=['ddG', 'dTm'], how='all')


    # drop rows missing any of the following columns: pdb_id, position, uniprot_id, wild_type, mutation
    df = df.dropna(subset=['uniprot_id', 'pdb_id', 'position', 'wild_type', 'mutation'], how='any')
    print('After dropping rows with missing values:', df.shape)

    # reconcile multiple PDB IDs
    df = reconcile_multi_pdbs(df, correction_file)

    # manually excluded proteins
    df = df.loc[df['pdb_id_corrected'] != '1BF4']
    df = df.loc[df['pdb_id_corrected'] != '1C8C']
    df = df.loc[df['pdb_id_corrected'] != '1FEP']

    # drop duplicates using new cleaned PDBs
    df = drop_duplicate_entries(df).reset_index(drop=True)
    print('Shape after dropping silent duplicate measurements:', df.shape)

    df['pdb_id_corrected'][df['pdb_id_corrected'].str.contains('1AEP')] = '1AEP'  # weird bug sometimes occurs due to excel auto-formatting
    # output cleaned dataset
    print('Cleaned dataset shape:', df.shape)
    df.to_csv(out_file)

    return

infile = '../data/fireprot/0_fireprotDB_raw.csv'
outfile = '../data/fireprot/1_fireprotDB_cleaned.csv'
correctionfile = '../data/fireprot/fireprotDB-pdbID-corrections.csv'

clean_fireprot_dataset(in_file=infile,
                       out_file=outfile,
                       correction_file=correctionfile)
