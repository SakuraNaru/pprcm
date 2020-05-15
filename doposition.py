import pandas as pd
import numpy as np
import re
import os
import sys
import threading

from multiprocessing import Pool
from multiprocessing import Queue


def readPairs(file_name):
    mix_name=[]
    protein_label=[]
    rna_label=[]

    with open(file_name) as f:
        for line in f.readlines():
            line=line.replace('\n','')
            line_s=line.split(" ")
            line_n=line_s[0].split(":")
            mix_name.append(line_n[0])
            protein_label.append(line_n[1])
            rna_label.append(line_s[1].split(":")[1])

    return mix_name,protein_label,rna_label

def findPos(name,dataset_name):
    # name=name.lower()
    # files=os.listdir('%s/cif_file'%dataset_name)
    # exist=False
    # file_name=''
    # for file in files:
    #     if re.search(name,file)!=None:
    #         exist=True
    #         file_name=file
    # if not exist:
    #     print('No such file ! Please download %s' %file)
    #     return None

    dataFrame=[]
    with open('atomfile/%s.cif'%name) as f:
        lines=f.readlines()
        for line in lines:
            if (re.match('ATOM',line)!=None):

                #deal with pdf file
                # residue_name=line[17:20].strip()
                # chain_id=line[21].strip()
                # residue_id=line[22:26].strip()
                # x=line[30:38].strip()
                # y=line[38:46].strip()
                # z=line[46:54].strip()

                #deal with cif file
                line=line.replace('\n','')
                s=line.split()
                residue_name=s[5]
                chain_id=s[18]
                residue_id=s[8]
                x=('%.2f'%float(s[10]))
                y=('%.2f'%float(s[11]))
                z=('%.2f'%float(s[12]))
                dataFrame.append([residue_name,chain_id,residue_id,x,y,z])

    df=pd.DataFrame(np.array(dataFrame),columns=['residue_name','chain_id','residue_id','x','y','z'])
    df[['residue_id','x','y','z']] = df[['residue_id','x','y','z']].apply(pd.to_numeric, errors='ignore')
    return df

def calDisMatrix(protein_df,rna_df):
    distance_matrix=[]
    protein_gb=protein_df.loc[:, ['x', 'y', 'z']].groupby(protein_df['residue_id'])
    rna_gb=rna_df.loc[:, ['x', 'y', 'z']].groupby(rna_df['residue_id'])

    for i in protein_gb:
        protein=np.array(i[1])
        for j in rna_gb:
            rna=np.array(j[1])

            distance=sys.float_info.max
            for atom in protein:
                dis_tmp=np.sqrt(np.sum(np.square(rna - atom), axis=1)).min()
                if dis_tmp<distance:
                    distance=dis_tmp
            distance_matrix.append(distance)

    distance_matrix=np.reshape(distance_matrix,[len(protein_gb),-1])

    return distance_matrix

def toSequence(df,type): #type=0---protein type=1----rna
    word_dict = {'ALA': 'A', 'PHE': 'F', 'CYS': 'C', 'ASP': 'D',
                 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
                 'HIS': 'H', 'LEU': 'L', 'ILE': 'I', 'LYS': 'K',
                 'MET': 'M', 'PRO': 'P', 'ARG': 'R', 'SER': 'S',
                 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    sequence=[]
    if type==0: #protein
        protein_gb=np.array(df['residue_name'].groupby(df['residue_id']).head(1))
        for i in protein_gb:
            if i in word_dict:
                sequence.append(word_dict[i])
    if type==1:
        rna_gb = np.array(df['residue_name'].groupby(df['residue_id']).head(1))
        for i in rna_gb:
            if i in {'A','U','G','C'}:
                sequence.append(i)
    return sequence

def writeSeq(dataset_name,seq,name,label,type): #0 for protein, 1 for rna
    if not os.path.exists('%s/pssm/input'%dataset_name):
        os.makedirs('%s/pssm/input/protein'%dataset_name)
        os.makedirs('%s/pssm/input/rna'%dataset_name)
    if type==0:
        path='%s/pssm/input/protein'%dataset_name
    if type==1:
        path='%s/pssm/input/rna'%dataset_name

    list = os.listdir(path)
    new_filename='%s_%s'%(name,label)
    if new_filename in list:
        return
    with open('%s/%s'%(path,new_filename), 'w') as f:
        f.writelines('>%s:%s|PDBID|CHAIN|SEQUENCE\n'%(name,label))
        num=0
        for i in seq:
            f.write(i)
            num+=1
            if num==60:
                num=0
                f.write('\n')
        f.write('\n')

def dociffile(dataset_name,mix_name, protein_label, rna_label):
    df = findPos(mix_name, dataset_name)
    protein_df = df[df['chain_id'].isin([str(protein_label)])]
    rna_df = df[df['chain_id'].isin([str(rna_label)])]
    if rna_df.empty | protein_df.empty:
        print('%s_%s+%s_%s not exist! Check cif!' % (
        mix_name, protein_label, mix_name, rna_label))
        # result_que.put('%s:%s %s:%s'%(mix_name, protein_label, mix_name, rna_label))
        return '%s:%s %s:%s'%(mix_name, protein_label, mix_name, rna_label)

    matrix = calDisMatrix(protein_df, rna_df)

    pos = np.where(matrix < 3)
    contact_map = np.zeros([np.shape(matrix)[0], np.shape(matrix)[1]])
    for i in range(len(pos[0])):
        contact_map[pos[0][i], pos[1][i]] = 1

    #NEW contact_map same as softmax
    # pos_neg = np.where(matrix >= 5)
    # contact_map_new = np.zeros([np.shape(matrix)[0], np.shape(matrix)[1],2])
    # for i in range(len(pos[0])):
    #     contact_map_new[pos[0][i], pos[1][i],1]=1
    # for i in range(len(pos_neg[0])):
    #     contact_map_new[pos_neg[0][i], pos_neg[1][i],0]=1

    num=np.sum(contact_map,dtype=np.int32)
    # print('%s_%s+%s_%s have %s' % (
    #     mix_name, protein_label, mix_name, rna_label,num))
    if num==0:
        print('%s_%s+%s_%s do not have contact!' % (
            mix_name, protein_label, mix_name, rna_label))
        return '%s:%s %s:%s'%(mix_name, protein_label, mix_name, rna_label)

    pro_seq = toSequence(protein_df, 0)
    rna_seq = toSequence(rna_df, 1)

    if (np.shape(matrix)[0]!=len(pro_seq))|(np.shape(matrix)[1]!=len(rna_seq)):
        print('%s_%s+%s_%s length no same!'% (mix_name, protein_label, mix_name, rna_label))
        # result_que.put('%s:%s %s:%s' % (mix_name, protein_label, mix_name, rna_label))
        return '%s:%s %s:%s'%(mix_name, protein_label, mix_name, rna_label)

    np.save('%s/distance_matrix/%s_%s+%s_%s.npy' % (dataset_name, mix_name, protein_label, mix_name, rna_label),
            contact_map)
    writeSeq(dataset_name, pro_seq, mix_name.upper(), protein_label, 0)
    writeSeq(dataset_name, rna_seq, mix_name.upper(), rna_label, 1)

    return None

if __name__ == '__main__':
    del_line = []
    jobs=[]
    mquelock = threading.Lock()
    # result_que=Queue()
    dataset_name='dataset2'

    if not os.path.exists('%s/distance_matrix' % dataset_name):
        os.mkdir('%s/distance_matrix' % dataset_name)

    with Pool(processes=os.cpu_count()) as pool:
        mix_name, protein_label, rna_label = readPairs('%s/redundant_pairs.txt' % dataset_name)
        for i in range(len(mix_name)):
            print(i)
            p=pool.apply_async(dociffile,(dataset_name, mix_name[i], protein_label[i], rna_label[i]))
            jobs.append(p)
            # p.wait()
        for i in jobs:
            # i.wait()
            result=i.get()
            if result!=None:
                del_line.append(result)


    # while not result_que.empty():
    #     del_line.append(result_que.get())

    with open('%s/redundant_pairs.txt' % dataset_name) as f:
        lines = f.readlines()
        for l in del_line:
            lines.remove(l+'\n')
        with open('%s/final_pairs.txt' % dataset_name, 'w') as w:
            w.writelines(lines)