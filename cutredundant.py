import re
import numpy as np
import os
def readCluster(path):
    dict = {}
    with open(path) as f:
        lines=f.readlines()
        split_position = []
        for i in range(len(lines)):
            lines[i]=lines[i].replace('\n','')
            if re.match('>', lines[i]):
                split_position.append(i)

        for pos in range(len(split_position)):
            cluster_no = lines[split_position[pos]].split(' ')[1]
            if pos < len(split_position) - 1:
                for seq_pos in range(split_position[pos] + 1, split_position[pos + 1]):
                    name=lines[seq_pos].split(' ')[1].split('|')[0][1:]
                    dict[name]=cluster_no
            else:
                for seq_pos in range(split_position[pos] + 1, len(lines)):
                    name=lines[seq_pos].split(' ')[1].split('|')[0][1:]
                    dict[name] = cluster_no
    return dict

def readPairs():
    protein_dict=readCluster('%s/cluster/protein.out.clstr'%dataset_name)
    rna_dict=readCluster('%s/cluster/rna.out.clstr'%dataset_name)
    cluster_set=set()
    with open('%s/first_pairs.txt'%dataset_name) as f:
        lines=f.readlines()
        delete_list=[]
        for i in range(len(lines)):
            lines[i]=lines[i].replace('\n','')
            protein,rna=lines[i].split(' ')
            try:
                protein_seq=protein_dict[protein]
                rna_seq=rna_dict[rna]
            except Exception:
                print('Wrong with %s!'%lines[i])
                delete_list.append(i)
                continue
            pair=(protein_dict[protein],rna_dict[rna])
            if pair in cluster_set:
                delete_list.append(i)
            else:
                cluster_set.add(pair)
        lines=np.delete(lines,delete_list)
        with open('%s/redundant_pairs.txt'%dataset_name,'w') as w :
            for i in lines:
                w.writelines(i+'\n')

def readNewPairs():
    protein_dict = readCluster('%s/cluster/protein.out.clstr' % dataset_name)
    rna_dict = readCluster('%s/cluster/rna.out.clstr' % dataset_name)
    cluster_set=set()
    with open('%s/first_pairs.txt' % dataset_name) as f:
        lines = f.readlines()
        delete_list = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            protein, rna = lines[i].split(' ')
            try:
                protein_seq = protein_dict[protein]
                rna_seq = rna_dict[rna]
            except Exception:
                print('Wrong with %s!' % lines[i])
                delete_list.append(i)
                continue
            pair = (protein_dict[protein], rna_dict[rna])
            if pair in cluster_set:
                delete_list.append(i)
            else:
                cluster_set.add(pair)
            # if protein_dict[protein] in protein_cluster:
            #     delete_list.append(i)
            # else:
            #     protein_cluster.add(protein_dict[protein])
            # if rna_dict[rna] in rna_cluster:
            #     delete_list.append(i)
            # else:
            #     rna_cluster.add(rna_dict[rna])
        lines = np.delete(lines, delete_list)
        with open('%s/redundant_pairs.txt' % dataset_name, 'w') as w:
            for i in lines:
                w.writelines(i + '\n')
def command():
    # os.system('~/cdhit/cd-hit -i ~/prpcm/%s/cluster/protein.txt -o ~/prpcm/%s/cluster/protein.out -l 9 -c 0.6 -n 4'%(dataset_name,dataset_name))
    # os.system('~/cdhit/cd-hit-est -i ~/prpcm/%s/cluster/rna.txt -o ~/prpcm/%s/cluster/rna.out -l 9 -c 0.8 -n 5'%(dataset_name,dataset_name))
    #windows
    # os.system('cdhit/cd-hit -i %s/cluster/protein.txt -o %s/cluster/protein.out -l 9 -c 1 -n 5' % (dataset_name, dataset_name))
    # os.system('cdhit/cd-hit-est -i %s/cluster/rna.txt -o %s/cluster/rna.out -l 9 -c 1 -n 8' % (dataset_name, dataset_name))
    os.system('cdhit/cd-hit -i %s/cluster/protein.txt -o %s/cluster/protein.out -l 9 -c 0.9 -n 5' % (dataset_name, dataset_name))
    os.system('cdhit/cd-hit-est -i %s/cluster/rna.txt -o %s/cluster/rna.out -l 9 -c 0.9 -n 9' % (dataset_name, dataset_name))
if __name__ == '__main__':
    # readCluster()
    dataset_name='dataset2'
    # command()
    readNewPairs()