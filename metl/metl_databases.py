

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,re

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


def im2html(url):
    return '<img src="' + url + '" width="400" >'

def save_figs(dms_ids,prefix,figure_function):
    outdir=os.path.join('databases',prefix)
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")
    urls_html=[]
    targetdir=os.path.join('databases','ProteinGym_substitutions')
    for dms_id in dms_ids:
        dms_filename=os.path.join(targetdir,f'{dms_id}.csv')
        df=pd.read_csv(dms_filename)
        fig,ax=plt.subplots(1,1,figsize=(6.4, 4))
        ax=figure_function(df,ax)
        im_filename=os.path.join(os.getcwd() ,outdir,f"{dms_id}.png")
        fig.savefig(im_filename)
        urls_html.append(im2html(im_filename))
    return urls_html
def histogram_func(df,ax):
    ax=df['DMS_score'].hist(bins=100,alpha=0.3,ax=ax)
    ax.set_xlabel('DMS_score')
    return ax
def mutant_func(df,ax):
    df['nb_mutant']=df['mutant'].apply(lambda x : len(re.findall(r":",x))+1)
    ax=df['nb_mutant'].hist(alpha=0.3,color='red',ax=ax)
    ax.set_xlabel('nb_mutant')
    return ax

def parity_plot_func(df,ax):
    df['nb_mutant'] = df['mutant'].apply(lambda x: len(re.findall(r":", x)) + 1)
    ax=df.plot.scatter(x='DMS_score',y='nb_mutant',alpha=0.05,ax=ax,color='green')
    return ax
def all_values_increasing_order(df):
    return df.sort_values(by=['DMS_total_number_mutants'])
def only_doubles_and_above_increasing_order(df):
    return df[df['includes_multiple_mutants']].sort_values(by=['DMS_total_number_mutants'])
def table_compare_datasets(filter_name,filter_func=all_values_increasing_order):
    ref_filename=os.path.join('databases','protein_gym_reference.csv')
    df=pd.read_csv(ref_filename).set_index('DMS_id')
    df=filter_func(df).copy()

    funcs2include=[('histogram',histogram_func),
                   ('mutant',mutant_func),
                   ('parity_plot',parity_plot_func)]
    for prefix, func2include in funcs2include:
        df[prefix]=save_figs(df.index,prefix,func2include)

    df['region_length'] = df['region_mutated'].apply(lambda x: abs(eval(x.replace('–', '-'))))
    xcols=['molecule_name','region_length','region_mutated','seq_len','DMS_total_number_mutants',
           'DMS_number_single_mutants', 'DMS_number_multiple_mutants',
           'year','source_organism','selection_assay','selection_type',
           'MSA_len','MSA_bitscore','MSA_num_seqs','taxon']

    xcols=[f[0] for f in funcs2include]+ xcols
    df=df[xcols].copy()
    df.to_html(os.path.join('databases',f"{filter_name}.html"),escape=False)

if __name__ == '__main__':
    # analyze_mavedb()
    # analyze_protein_gym()
    # table_compare_datasets('all_datasets_all_xcols')
    table_compare_datasets('only_doubles_and_above_increasing',only_doubles_and_above_increasing_order)