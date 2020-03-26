# Import MPI
from mpi4py import MPI
import numpy as np

# Get comm object
comm = MPI.COMM_WORLD

# Process rank
rank = comm.Get_rank()

if rank == 0:

    # Can send various types of data
    d1 = 1
    d2 = "Hello World"
    d3 = [1, 2 , 3]
    d4 = {"Hello": "World"}

    # Blocking Communication
    comm.send(d1, 1, 1)
    comm.send(d2, 1, 2)
    comm.send(d3, 1, 3)
    
    # Non-blocking Communication
    req = comm.isend(d4, 1, 4)
    req.wait()

    # Faster way of communication
    d5 = np.arange(100, dtype=np.float64)

    # Note use of capitalised methods for faster communications
    comm.Send(d5, 1, 5)

elif rank == 1:
    d1 = comm.recv(source=0, tag=1)
    d2 = comm.recv(source=0, tag=2)
    d3 = comm.recv(source=0, tag=3)
    
    req = comm.irecv(source=0, tag=4)
    d4 =req.wait()

    d5 = np.empty(100, dtype=np.float64)
    comm.Recv(d5, 0, 5)

    print(d1, d2, d3 ,d4)
    print(d5)