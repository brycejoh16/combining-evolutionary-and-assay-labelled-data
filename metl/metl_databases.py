

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_mavedb():
    filename=os.path.join('datasets','mavedb_top_performers.csv')
    # this is the dataset which had >2 mutations and >1e4 datapoints
    df=pd.read_csv(filename).set_index('target_urn')
    df_filtered=df[df['nb_mut_greater_than_2']>0]
    df_filtered= df_filtered.sort_values(by=['nb_mut_greater_than_2'])

    df_filtered['target_name'][df_filtered['target_name']=='SARS-CoV-2 receptor binding domain']='SARS-CoV-2 bind recp'
    print(df_filtered)
    fig1,ax1=plt.subplots(1,1,constrained_layout=True)
    fig,ax=plt.subplots(1,2)

    ax1=df_filtered.plot.scatter(logy=True,x='target_name',y='nb_mut_greater_than_2',
                                 c='percentage_greater_than_2',cmap='plasma',ax=ax1,rot=90)
    df_filtered.hist(column='nb_mut_greater_than_2',bins=20,ax=ax[0])
    df_filtered.hist(column='percentage_greater_than_2',bins=20,ax=ax[1])
    fig.suptitle(f'mavedb for nb variants >1e4 and 2+ mutations'
                 f'nb variants {len(df_filtered)}')
    ax1.set_title(f'mavedb: {len(df_filtered["nb_mut_greater_than_2"].unique())} unique datasets,'
                  f'nb variants >1e4 and 2+ mutations')
    ax1.grid(axis='x')
    fig1.savefig(os.path.join('datasets','mavedb_colorbar.png'))
    fig.savefig(os.path.join('datasets','mavedb_hist.png'))


def analyze_protein_gym():
    filename=os.path.join('datasets','protein_gym_reference.csv')
    df=pd.read_csv(filename)
    print(f'nb of datasets that include multiple mutants\n'
         f'{df["includes_multiple_mutants"].sum()} out of '
          f'{len(df)}')
    df_filtered=df[df['includes_multiple_mutants']].copy()
    df_filtered = df_filtered.sort_values(by=['DMS_number_multiple_mutants'])
    print(df_filtered)
    fig1, ax1 = plt.subplots(1, 1,constrained_layout=True)
    fig, ax = plt.subplots(1, 2)
    df_filtered['ratio_mut_to_total']=df['DMS_number_multiple_mutants']/df['DMS_total_number_mutants']
    ax1=df_filtered.plot.scatter(logy=True,x='UniProt_ID',y='DMS_total_number_mutants',
                             c='ratio_mut_to_total',cmap='plasma',ax=ax1,rot=90)
    # fig1.tight_layout()
    ax1.set_title(f'protein_gym: {len(df_filtered)} unique datasets with 2+ mutations')
    ax1.grid(axis='x')


    df_filtered.hist(column='DMS_number_multiple_mutants', bins=20, ax=ax[0])
    df_filtered.hist(column='ratio_mut_to_total', bins=20, ax=ax[1])
    ax1.set_title('protein_gym for datasets with 2+ mutants , '
                 f' tot: {len(df_filtered)}')

    fig1.savefig(os.path.join('datasets', 'protein_gym_colorbar.png'))
    fig.savefig(os.path.join('datasets', 'protein_gym_hist.png'))

if __name__ == '__main__':
    analyze_mavedb()
    analyze_protein_gym()