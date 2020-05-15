import os
import re
import sys
from doposition import findPos
from doposition import toSequence
import numpy as np
def readFile(dataset_name,protein_min=0,protein_max=sys.maxsize,rna_min=0,rna_max=sys.maxsize): #dataset_name represent for the name of dataset
    file_list = os.listdir('%s/seq_file' % dataset_name)
    for file in file_list:
        with open('%s/seq_file/%s' % (dataset_name, file)) as f:
            lines = f.readlines()
            split_position = []
            protein_name=[]
            rna_name=[]
            for i in range(len(lines)):
                lines[i] = lines[i].replace('\n', '')
                if re.match('>', lines[i]) != None:
                    split_position.append(i)
            for pos in range(len(split_position)):
                name = lines[split_position[pos]].split('|')[0].replace('>', '')
                seq = ''
                if pos < len(split_position) - 1:
                    for seq_pos in range(split_position[pos] + 1, split_position[pos + 1]):
                        seq += lines[seq_pos]
                else:
                    for seq_pos in range(split_position[pos] + 1, len(lines)):
                        seq += lines[seq_pos]
                if isProtein(seq):
                    if (len(seq) <= protein_min) | (len(seq) > protein_max):
                        continue
                    path='protein.txt'
                    protein_name.append(name)
                else:
                    if (len(seq) <= rna_min) | (len(seq) > rna_max):
                        continue
                    path='rna.txt'
                    rna_name.append(name)
                if not os.path.exists('%s/cluster' %  dataset_name):
                    os.mkdir('%s/cluster' %  dataset_name)
                with open('%s/cluster/%s' %  (dataset_name,path),'a+') as w:
                    w.writelines('>'+name+'\n')
                    w.writelines(seq+'\n')

            if len(protein_name)==0:
                print('protein is NULL!')
            elif len(rna_name)==0:
                print('rna is NULL!')
            else:
                with open('%s/first_pairs.txt'%dataset_name, 'a+') as w:
                    for i in protein_name:
                        for j in rna_name:
                            w.writelines('%s %s\n'%(i,j))

def isProtein(seq):
    dict = ['F', 'D', 'N', 'E', 'Q', 'H', 'L', 'I', 'K',
            'M', 'P', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in seq:
        if i in dict:
            return True
    return False
def readActSequence(dataset_name):
    with open('%s/finaldownload.txt'%dataset_name) as f:
        line=f.readline()
        download_names=line.split(', ')

    for name in download_names:
        protein_name = []
        rna_name = []
        try:
            df=findPos(name,dataset_name)
        except Exception:
            print('%s get something wrong!'%name)
            continue

        chain_gb=df.loc[:,['residue_name','chain_id','residue_id']].groupby(df['chain_id'])
        for chain in chain_gb:
            label=chain[0]
            atom=chain[1]

            res=np.array(atom.head(1))[0,0]
            if len(res)>1: #protein
                type=0
                path='protein.txt'
            else: #rna
                type = 1
                path = 'rna.txt'
            seq = toSequence(atom, type)
            seq=''.join(seq)
            if (len(seq)>500)|(len(seq) < 10):
                continue

            if len(res)>1: #protein
                protein_name.append('%s:%s'%(name,label))
            else: #rna
                rna_name.append('%s:%s' % (name, label))

            if not os.path.exists('%s/cluster' % dataset_name):
                os.mkdir('%s/cluster' % dataset_name)
            with open('%s/cluster/%s' % (dataset_name, path), 'a+') as w:
                w.writelines('>%s:%s|PDBID\n'%(name,label))
                w.writelines(seq + '\n')
        if len(protein_name) == 0:
            print('protein is NULL!')
        elif len(rna_name) == 0:
            print('rna is NULL!')
        else:
            with open('%s/first_pairs.txt' % dataset_name, 'a+') as w:
                for i in protein_name:
                    for j in rna_name:
                        w.writelines('%s %s\n' % (i, j))

if __name__ == '__main__':
    # readFile('dataset1')
    readActSequence('dataset1')
