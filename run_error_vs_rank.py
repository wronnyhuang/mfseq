from multiprocessing import Process
import subprocess
import numpy as np

def run(rank, gpu):
  command = 'python main.py -rank=%s -gpu=%s' % (rank, gpu)
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

nproc = 8 # number of parallel processes to run at any given time. e.g. if 8, and you have 4 gpus, then 2 procs on each gpu
ngpu = 4
processes = []
for rank in range(1, 400):

  i = 0
  while True:  # cycle iterate through the length of the list (nproc)

    # keep iterate below nproc as it cycles
    i = np.mod(i, nproc)

    # if list hasn't been built up yet (first nproc iterations)
    if len(processes) < nproc:
      gpu = np.mod(len(processes), ngpu)
      process = Process(target=run, args=[rank, gpu])
      process.start()
      processes = processes + [process]
      break

    # check if process is done; if not, then increment iterate; if so, start new process
    process = processes[i]
    if process.is_alive(): i += 1; continue
    del(process)

    # start new process
    gpu = np.mod(i,3)
    process = Process(target=run, args=[rank, gpu])
    process.start()
    processes[i] = process
    break

