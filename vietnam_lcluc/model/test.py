import numpy as np
import time

a = np.random.randint(1, size=(20000, 20000, 4))
print(a[0,0,0])
start = time.process_time()
print(np.all(a == a[0,0,0]))
print(time.process_time() - start)

#>>> np.any(a != 3)
#True
#>>> np.any(a == 3)
