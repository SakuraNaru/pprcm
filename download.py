import os
import requests
import threading
import queue


def find_name(path):
    names = []
    for file in os.listdir(path):
        names.append(file.split('.')[0].lower())
    return names


class mThread(threading.Thread):
    def __init__(self,que):
        threading.Thread.__init__(self)
        self.que = que
    def run(self):
        while 1:
            if not self.que.empty():
                que_lock.acquire()
                name = self.que.get()
                que_lock.release()

                try:
                    download(name)
                except Exception as e:
                    print("retry to download %s" % name)
                    que_lock.acquire()
                    self.que.put(name)
                    que_lock.release()
                    continue

                que_lock.acquire()
                print("%s has been done" %name)
                que_lock.release()
            else:
                break


def download(name):
    print("now download %s" % name)
    url = "https://files.rcsb.org/download/%s.cif" % name
    data = requests.get(url)
    with open("atomfile/%s.cif" % name, "wb") as code:
        code.write(data.content)


def getName(path):
    with open(path) as f:
        list=f.readlines()
        name=list[0].replace('\n','').split(', ')
    return name


if __name__ == '__main__':
    dataset_name='dataset1'
    names= getName('%s/finaldownload.txt'%dataset_name)

    exist=os.listdir('atomfile')
    for i in exist:
        name=i.split('.')[0]
        if name in names:
            names.remove(name)

    que_lock=threading.Lock()

    que = queue.Queue()

    que_lock.acquire()
    for name in names:
        que.put(name)
    que_lock.release()

    for i in range(100):
        thread = mThread(que)
        thread.start()
