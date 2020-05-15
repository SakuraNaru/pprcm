import threading
import os
import queue


class mThread(threading.Thread):
    def __init__(self, que, is_protein,dataset_name):
        threading.Thread.__init__(self)
        self.que = que
        self.is_protein = is_protein
        self.dataset_name=dataset_name

    def run(self):
        while not self.que.empty():
            queueLock.acquire()
            name = self.que.get()
            queueLock.release()
            print('making %s pssm!' % name)

            if self.is_protein:
                # os.system('psiblast -query %s/pssm/input/protein/%s \
                #           -db ~/blastdb/nr \
                #           -num_iterations 3 \
                #           -out pssmfile/out/%s.out\
                #           -out_ascii_pssm pssmfile/protein/%s.pssm ' % (self.dataset_name,name, name,name))
                os.system('/mnt/d/ncbi-blast-2.7.1+/bin/psiblast -query %s/pssm/input/protein/%s\
                                          -db /mnt/d/ncbi-blast-2.7.1+/db/swiss \
                                          -num_iterations 3 \
                                          -out pssmfile/out/%s.out\
                                          -out_ascii_pssm pssmfile/protein/%s.pssm ' % (self.dataset_name, name, name, name))
            else:
                os.system('psiblast -query %s/pssm/input/rna/%s \
                            -db ~/blastdb/nt \
                            -num_iterations 3 \
                            -out %s/pssm/out/rna/%s.out\
                            -out_ascii_pssm %s/pssm/outpssm/rna/%s.pssm ' % (self.dataset_name,name, self.dataset_name,name, self.dataset_name,name))
            print('%s pssm has done' % name)

def makepssm(dataset_name,protein_list):
    # dataset_name = 'dataset9'

    if not os.path.exists('%s/pssm/out' % dataset_name):
        os.makedirs('%s/pssm/out/protein' % dataset_name)
        os.makedirs('%s/pssm/outpssm/protein' % dataset_name)
    if not os.path.exists('pssmfile'):
        os.makedirs('pssmfile/protein')
        os.makedirs('pssmfile/out')
        os.makedirs('pssmfile/rna')

    exist = os.listdir('pssmfile/protein')
    for i in exist:
        name = i.split('.')[0]
        if name in protein_list:
            protein_list.remove(name)

    print('%s to make!' % len(protein_list))
    que = queue.Queue()
    global queueLock
    queueLock = threading.Lock()
    queueLock.acquire()
    for i in protein_list:
        que.put(i)
    queueLock.release()

    for x in range(16):
        thread = mThread(que, True, dataset_name)
        thread.start()


if __name__ == '__main__':
    dataset_name='dataset1'
    protein = []
    with open('%s/final_pairs.txt' % dataset_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line = line.replace(':', '_')
            ls = line.split(' ')
            protein.append(ls[0])
    protein = list(set(protein))

    makepssm('dataset1',protein)