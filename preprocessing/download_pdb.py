import os
import sys
from urllib import request
import pandas as pd
import numpy as np


def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/", assembly=-1):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    # downloadurl = "https://files.rcsb.org/pub/pdb/data/biounit/PDB/all"
    # downloadurl = "https://files.wwpdb.org/pub/pdb/data/biounit/PDB/all"

    if assembly == -1:
        suffix = ".pdb"
    else:
        suffix = ".pdb" 
    pdbfn = pdbcode + ".pdb7"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def main():
    """Load many PDB codes and download them all from RCSB PDB"""
    # check data dir and make if needed
    data_dir = '/Users/henry/Documents/kuhlman-lab/fireprotDB/pdb_files'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    # load pdb codes
    df = pd.read_csv('/Users/henry/Documents/kuhlman-lab/fireprotDB/1_cleaned_fireprotDB.csv',
                     header=0, usecols=['pdb_id_corrected', 'uniprot_id'], dtype='str')

    codes = np.unique([c[1] for c in df.values])
    print(list(codes))
    # sanity check to make sure PDB ID count matches Uniprot ID count
    assert df['pdb_id_corrected'].unique().size == df['uniprot_id'].unique().size

    # cycle through PDB codes and download all of them
    for code in codes:
        print(code)
        for assembly in range(1, 10):
            download_pdb(code, '/Users/henry/Documents/kuhlman-lab/fireprotDB', assembly)


main()
