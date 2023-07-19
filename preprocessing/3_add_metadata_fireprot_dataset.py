import os
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm


def main(csv_loc, meta_loc, pdb_loc):
    # load validated csv
    df = pd.read_csv(csv_loc)
    pdb_ids = df['pdb_id_corrected'].unique() # grab pdb ids from csv

    # load oligomeric state metadata csv
    odf = pd.read_csv(meta_loc)

    # assign each PDB its metadata
    df['oligomeric_state'] = ''
    df['structure_method'] = ''
    df['resolution'] = 0
    
    for pid in tqdm(pdb_ids):
        # get PDB metadata
        fname = os.path.join(pdb_loc, pid + '.pdb')
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        struct = parser.get_structure('tmp', fname)
        header = struct.header
        # print('PDB:', pid, 'HEADER:', header['resolution'], header['structure_method'])
        # add metadata to CSV dataframe
        df.loc[df['pdb_id_corrected'] == pid, 'structure_method'] = header['structure_method']
        df.loc[df['pdb_id_corrected'] == pid, 'resolution'] = header['resolution']
        try:
            df.loc[df['pdb_id_corrected'] == pid, 'oligomeric_state'] = odf[odf['pdb_id'] == pid]['oligomeric_state'].values[0]
        except IndexError:  # need to handle weird broken cases from Excel formatting bugs
            # print(pid)
            if pid == '1E21':
                # print(odf[odf['pdb_id'] == '1.00E+21'])
                df.loc[df['pdb_id_corrected'] == pid, 'oligomeric_state'] = \
                odf[odf['pdb_id'] == '1.00E+21']['oligomeric_state'].values[0]
            elif pid == '1DEC':
                # print(odf[odf['pdb_id'] == '1-Dec'])
                df.loc[df['pdb_id_corrected'] == pid, 'oligomeric_state'] = \
                odf[odf['pdb_id'] == '1-Dec']['oligomeric_state'].values[0]
        # print('*' * 50)

    return df

csv_loc = '../data/fireprot/2_fireprotDB_aligned.csv'
meta_loc = '../data/fireprot/fireprotDB-curation-metadata.csv'
pdb_loc = '../data/fireprot/pdbs/monomers'

df = main(csv_loc, meta_loc, pdb_loc)

out_loc = '../data/fireprot/3_fireprotDB_wmetadata.csv'
df.to_csv(out_loc)
