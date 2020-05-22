import numpy as np

no_chunks = int(1e8/1000000)
chunk_list = []
total = 10000000
for i in range(no_chunks-1):
    val = np.random.randint(0, total)
    chunk_list.append(val)
    total -= val
chunk_list.append(total)
print(chunk_list)