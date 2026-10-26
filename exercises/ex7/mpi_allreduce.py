import os
import socket
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

print(f"Python MPI Rank {rank:02d}/{size:02d} online on {host}")
comm.Barrier()

# Create a synthetic gradient tensor (1 million floats = 4 MB per rank)
tensor_size = 1_000_000
send_data = np.ones(tensor_size, dtype=np.float32) * (rank + 1)
recv_data = np.zeros(tensor_size, dtype=np.float32)

t_start = MPI.Wtime()
comm.Allreduce(send_data, recv_data, op=MPI.SUM)
t_end = MPI.Wtime()

# Expected sum formula: n * (n + 1) / 2
expected_value = (size * (size + 1)) / 2.0
is_correct = np.allclose(recv_data[0], expected_value)

comm.Barrier()
if rank == 0:
    elapsed_ms = (t_end - t_start) * 1000.0
    mb_transferred = (tensor_size * 4) / (1024 * 1024)
    print(f"\n[AllReduce Validation]")
    print(f"Ranks Participating : {size}")
    print(f"Tensor Payload Size : {mb_transferred:.2f} MB")
    print(f"AllReduce Duration  : {elapsed_ms:.2f} ms")
    print(f"Result Verification : {'PASSED' if is_correct else 'FAILED'}")
    print(f"Value at Index [0]  : {recv_data[0]} (Expected: {expected_value})\n")
