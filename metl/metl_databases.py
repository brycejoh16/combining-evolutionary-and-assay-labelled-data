

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

def check_sub_file_protein_gym():
    df=pd.read_csv(os.path.join('databases','reference.csv'))
    for row in df.itertuples():
        print(f"\n"+row.DMS_id)
        df_temp=pd.read_csv(os.path.join('databases','protein_gym_benchmark',row.DMS_filename))
        start,end=row.region_mutated.split('-')
        start,end=int(start),int(end)

        total_regoin=end-start+1 # add one since inclusive
        reduced_WT=row.target_seq[start-1:end]
        assert total_regoin>0 and total_regoin==len(reduced_WT), 'value should always be greater than zero'
        mutant_count=[]
        min=np.inf
        max=-np.inf
        lst_of_errors = []
        for i,mutants in enumerate(df_temp['mutant']):
            lst_mutants=mutants.split(':')
            mutant_count.append(len(lst_mutants))

            for mutant in lst_mutants:
                WTaa,pos,Mutaa=mutant[0],mutant[1:-1],mutant[-1]
                relative_pos=int(pos)-start
                try:
                    if WTaa != reduced_WT[relative_pos]:
                        lst_of_errors.append(f'\nmutant {mutant} was found in list but didnt match to wildtype')
                except Exception:
                    pass
                    # lst_of_errors.append(f'relative position {relative_pos} not within bounds')

                if int(pos) < start or int(pos) >end:
                    lst_of_errors.append(f'pos:{pos} less than start:{start} or greater than end:{end}')
                if relative_pos > max :
                    max=relative_pos

                if relative_pos < min:
                    min=relative_pos
        if len(lst_of_errors)>0:
            df_err=pd.DataFrame()
            df_err['errors']=lst_of_errors
            print(dict(df_err.value_counts()))

        if min!=0 :
            print(f'found min:{min+start}, expected: {start} error finding min for {row.DMS_id} ')
        if max!=end-start:
            print(f"found max:{max+start}, expected: {end} error finding the max for {row.DMS_id}")

        mutant_count=np.array(mutant_count)
        if (mutant_count==1).sum() != row.DMS_number_single_mutants:
            print(f"\nfound off balance for single mutants in {row.DMS_id}")
        if (mutant_count!=1).sum() != row.DMS_number_multiple_mutants:
            print(f"\nfound off balance for double mutants+ in {row.DMS_id}")

        if len(mutant_count)!= row.DMS_total_number_mutants :
            print(f"number of found mutants not the same in {row.DMS_id}")





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
def experimental_datasets_sars():
    filename=os.path.join('databases','experimental','SPIKE_SARS2_Starr_bind_2020_doubles')
    df=pd.read_csv(f"{filename}.csv")
    # remove 'NA'
    print(f'original length of sequences : {len(df)}')
    df=df[~np.isnan(df['score'])]
    assert np.isnan(df['score']).sum()==0
    print(f"after removing NA's :{len(df)}")
    df_new=df.groupby(['hgvs_pro'],as_index=False).mean()

    assert len(df_new['hgvs_pro'].unique())==len(df_new)

    df_new['nb_mutations']=df_new['hgvs_pro'].apply(lambda  x: len(x.split(";")))
    print(f"only looking at unique mutations: {len(df_new)}")

    ax=df_new.hist(column=['nb_mutations'],color='r',alpha=0.3)
    plt.savefig(f"{filename}_hist_nb_mutations.png")

    df_new.hist(column=['score'],bins=50,color='b',alpha=0.3)
    plt.savefig(f"{filename}_hist_score.png")

    df_new.plot.scatter(x='score',y='nb_mutations',color='g',alpha=0.2)
    plt.savefig(f"{filename}_parity_plot.png")

    print(f'number of double mutants: {(df_new["nb_mutations"]>1).sum()}')

    # do a group by for mutants, take the mean of the score

if __name__ == '__main__':
    # analyze_mavedb()
    # analyze_protein_gym()
    # table_compare_datasets('all_datasets_all_xcols')
    # table_compare_datasets('only_doubles_and_above_increasing',only_doubles_and_above_increasing_order)
    # check_sub_file_protein_gym()
    experimental_datasets_sars()