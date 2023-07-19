import os
from Bio.PDB import PDBParser
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np


def drop_duplicate_entries(tmp):
    """Drop cases where mutation and ddG/dTm value is the same, but other info is different"""
    arr = np.empty(tmp.shape[0], dtype=object)

    # get unique entries by ddG/dTm value
    for n, i in enumerate(tmp.index):
        arr[n] = str(tmp['pdb_id_corrected'][i]) + '-' + str(tmp['mutation'][i]) + \
                 '-' + str(tmp['pdb_position'][i]) + '-' + str(tmp['wild_type'][i]) + '-' \
                 + str(tmp['uniprot_id'][i]) + '-' + str(tmp['pH'][i]) + '-' + str(tmp['ddG'][i]) + str(tmp['dTm'][i])
    tmp['dupe_detector'] = arr
    # drop duplicate ddG/dTm values for same mutation(s) and conditions
    tmp = tmp.drop_duplicates(subset=['dupe_detector'])
    print('Shape after dropping explicit duplicate measurements:', tmp.shape)
    # remove all points with conflicting measurements at same pH
    ddG_only = tmp.dropna(subset=['ddG'], how='all')
    dTm_only = tmp.dropna(subset=['dTm'], how='all')
    df_list = []
    for case, case_col in zip([ddG_only, dTm_only], ['ddG', 'dTm']):
        row_list = []
        # get unique entries by mutation D only
        arr = np.empty(case.shape[0], dtype=object)
        for n, i in enumerate(case.index):
            arr[n] = str(case['pdb_id_corrected'][i]) + '-' + str(case['mutation'][i]) + \
                     '-' + str(case['pdb_position'][i]) + '-' + str(case['wild_type'][i]) + '-' \
                     + str(case['uniprot_id'][i])
        case['dupe_detector'] = arr

        for a in tqdm(case['dupe_detector'].unique()):
            rows = case[case['dupe_detector'] == a]
            if rows.shape[0] > 1:
                # print('Duplicate measurement by mutation ID')
                all_pH_values = [rows['pH'][r] for r in rows.index]
                pH_differences = np.array([round(abs(7.4 - a), 2) for a in all_pH_values])  # round to within 0.1 pH
                pH_differences[np.isnan(all_pH_values)] = 14  # treat NaN as largest possible pH gap
                # if all NaN, skip this
                lowest_pH_diff = min([p for p in pH_differences if p is not None])
                best_pH_values = [i for i in range(len(pH_differences)) if pH_differences[i] == lowest_pH_diff]
                # extract all scores from these rows
                best_rows_by_pH = rows.index[best_pH_values]
                scores = rows[case_col][best_rows_by_pH]
                # print(scores.unique())
                if len(scores.unique()) > 1:  # if multiple unique scores at best pH, drop all of them for now
                    avg_score = np.mean(scores)
                    # print('Average pH', avg_score)
                    rows[case_col][best_rows_by_pH[0]] = avg_score
                    # print(rows.loc[best_rows_by_pH[0]].shape, '%%%')
                    row_list.append(rows.loc[[best_rows_by_pH[0]]])
                else:  # otherwise, just keep best one
                    # print('One unique score at best pH; keeping just one:', best_rows_by_pH)
                    # print(rows.loc[best_rows_by_pH].shape)
                    row_list.append(rows.loc[best_rows_by_pH])
            else:
                row_list.append(rows)
        print('Rows retained:', len(row_list))
        df_list.append(pd.concat(row_list, axis=0))

    df_total = pd.concat(df_list, axis=0)
    print('Shape after pH consolidation:', df_total.shape)
    return df_total


def drop_missing_pH(tmp):
    """Drop cases where mutation and ddG/dTm value is the same, but other info is different"""
    pH_values = tmp['pH']
    valid_pH = [not np.isnan(p) for p in pH_values]
    print('Rows with valid pH values:', sum(valid_pH))
    new_tmp = tmp.loc[valid_pH]
    return new_tmp


def split_monomers_oligomers(tmp):
    monomers = df[df['oligomeric_state'] == 'monomer']
    oligomers = df[df['oligomeric_state'] != 'monomer']
    return monomers, oligomers


# df = pd.read_csv('/Users/henry/Documents/kuhlman-lab/fireprotDB/biounit_pdb_files/3_fireprotDB_cleaned_validated_wmetadata.csv')
df = pd.read_csv('../data/fireprot/3_fireprotDB_wmetadata.csv')
print(df.shape)

# split monomers and oligomers
monomers, oligomers = split_monomers_oligomers(df)
# monomers.to_csv('/Users/henry/Documents/kuhlman-lab/fireprotDB/v3_dataset_01042023/4A_fireprotDB_cleaned_monomers_allmutations.csv')
# oligomers.to_csv('/Users/henry/Documents/kuhlman-lab/fireprotDB/v3_dataset_01042023/4B_fireprotDB_cleaned_oligomers_allmutations.csv')

# keep only rows w pH values
# final_cleaned = drop_missing_pH(monomers)
# final_cleaned.to_csv('/Users/henry/Documents/kuhlman-lab/fireprotDB/v3_dataset_01042023/4C_fireprotDB_curated_valid_pH_only.csv')

# consolidate pH values
final_cleaned = drop_duplicate_entries(monomers)
final_cleaned = final_cleaned.loc[:, ~final_cleaned.columns.str.contains('^Unnamed')]
final_cleaned.to_csv('../data/fireprot/4_fireprotDB_bestpH.csv')
