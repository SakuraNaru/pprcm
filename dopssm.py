import pandas as pd
import numpy as np
import os
from sklearn.decomposition import TruncatedSVD

def readPSSM(file_name,protein_len):
    with open(file_name) as f:
        df=pd.read_csv(file_name,'\s+',skiprows=[0,1,2],header=None)
        df=df.iloc[0:protein_len,1:22]
        df.columns=['Atom','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        df=df.fillna(0)
    return df
def rnaOneHot(file_name):
    with open(file_name) as f:
        f.readline()
        seq=''
        onehot_array=[]
        for line in f.readlines():
            seq+=line.replace('\n','')

        dict={'A':[1,0,0,0],'U':[0,1,0,0],'G':[0,0,1,0],'C':[0,0,0,1]}
        # dict={'A':1,'U':2,'G':3,'C':4}
        for atom in seq:
            onehot_array.append(dict[atom])
    df=np.array(onehot_array)
    return df

def proteinOneHot(file_name):
    with open(file_name) as f:
        f.readline()
        seq=''
        for line in f.readlines():
            seq+=line.replace('\n','')

        dict=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

        onehot_array=oneHot(dict,seq)
        df=np.array(onehot_array)
        return df
def proteinEncoding(file_name):
    with open(file_name) as f:
        f.readline()
        seq=''
        for line in f.readlines():
            seq+=line.replace('\n','')

        onehot_array=[]
        letterDict = {}
        # dict=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
        letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
        letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
        letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
        letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
        letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
        letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
        letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
        letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
        letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
        letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
        letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
        letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
        letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
        letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
        letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
        letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
        letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
        letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
        letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
        # letterDict["X"] = [0, -0.00005, 0.00005, 0.0001, -0.0001, 0]
        for atom in seq:
            onehot_array.append(letterDict[atom])
        return onehot_array
def oneHot(alphabet,arr):
    char_to_int = dict((alphabet[i], i) for i in range(len(alphabet)))
    integer_encoded = [char_to_int[char] for char in arr]
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def makeInput(dataset_name):
    if not os.path.exists('%s/data'%dataset_name):
        os.makedirs('%s/data/input'%dataset_name)
        os.makedirs('%s/data/output'%dataset_name)
        os.makedirs('%s/data/label'%dataset_name)

    del_list = []
    lines=[]
    with open('%s/final_pairs.txt'%dataset_name) as f:
        x=[]
        y=[]
        label=[]
        iter=0
        # length=0
        name_no=0
        lines=f.readlines()
        for line in lines:
            iter += 1

            line = line.replace('\n', '')
            line = line.replace(':', '_')
            ls=line.split(' ')
            try:
                contactMap = readContactMap('%s/distance_matrix/%s+%s.npy' % (dataset_name, ls[0], ls[1]))
            except Exception:
                print('contact map %s+%s not exist!'%(ls[0], ls[1]))
                del_list.append(iter-1)
                continue

            contact_shape = np.shape(contactMap)
            # pos = np.where(contactMap == 1)
            # protein_atom = list(set(pos[0]))
            # rna_atom = list(set(pos[1]))
            # if (len(protein_atom)>int(contact_shape[0]/2))|(len(rna_atom)>int(contact_shape[1]/2)):
            #     print('too long!')
            #     continue
            try:
                protein_df = readPSSM('pssmfile/protein/%s.pssm'%ls[0],np.shape(contactMap)[0])
                # protein_array = proteinEncoding('%s/pssm/input/protein/%s'%(dataset_name, ls[0]))
                # protein_df = readSinglePssm(ls[0], np.shape(contactMap)[0], dataset_name)
                # protein_array = proteinOneHot('%s/pssm/input/protein/%s'%(dataset_name, ls[0]))
            except Exception:
                print('protein %s not exist!'%ls[0])
                del_list.append(iter - 1)
                continue

            try:
                rna_array = rnaOneHot('%s/pssm/input/rna/%s' % (dataset_name, ls[1]))
            except Exception:
                print('rna %s not exist!'%ls[1])
                del_list.append(iter - 1)
                continue

            protein_array = protein_df.iloc[:, 1:21].values

            # try:
            #     protein_array=np.concatenate((protein_array,protein_array_2),axis=1)
            # except:
            #     print('%s protein has different shape!' % (ls[0]))
            #     continue

            input_x = []
            for protein_atom in protein_array:
                for rna_atom in rna_array:
                    channel = np.dot(np.reshape(protein_atom, [-1, 1]), np.reshape(rna_atom, [1, -1]))
                    channel = channel.flatten()
                    # channel = np.concatenate([protein_atom,rna_atom])
                    input_x.append(channel)

            try:
                pssm = np.array(input_x).reshape([np.shape(protein_array)[0], np.shape(rna_array)[0], -1])
            except Exception:
                print('Something is wrong with %s+%s !'%(ls[0], ls[1]))
                del_list.append(iter - 1)
                continue

            pssm_shape = np.shape(pssm)

            if (pssm_shape[0] != contact_shape[0]) | (pssm_shape[1] != contact_shape[1]):
                print('%s and %s has different shape!'%(ls[0],ls[1]))
                del_list.append(iter - 1)
                continue
            x.append(pssm)
            y.append(contactMap)
            label.append('%s+%s'%(ls[0],ls[1]))
            # length += 1
            # a=np.shape(pssm)[:2]
            # b=np.shape(contactMap)[:2]
            # if a!=b:
            #     print("%s+%s,%s"%(np.shape(pssm),np.shape(contactMap),line))
            # print(len(x))
            if (len(x)==50)|(iter==len(lines)):
                np.save('%s/data/input/input_x_%s.npy'%(dataset_name,name_no), x)
                np.save('%s/data/output/output_y_%s.npy'%(dataset_name,name_no), y)
                np.save('%s/data/label/label%s.npy'%(dataset_name,name_no), label)
                name_no+=1
                # length=0
                x=[]
                y=[]
                label=[]
    with open('%s/final_pairs.txt' % dataset_name,'w') as w:
        new_list=np.delete(lines,del_list)
        for i in new_list:
            w.writelines(i)

def readContactMap(file_name):
    contact_map = np.load(file_name)
    # pos = np.where(matrix < 5)
    # contact_map = np.zeros([np.shape(matrix)[0], np.shape(matrix)[1]])
    # for i in range(len(pos[0])):
    #     contact_map[pos[0][i], pos[1][i]] = 1
    return contact_map
if __name__ == '__main__':
    # protein_df=readPSSM('1RY1_D.pssm')
    # rna_array=rnaOneHot('1RY1_M')
    makeInput('dataset2')