import json
import numpy as np

with open("data.json","r") as f:
    base = np.array(json.loads(f.read())["after"]["output"])
print(base.shape)
with open("result.json","r") as f:
    data = np.array(json.loads(f.read())["data"])
print(data.shape)
data = np.transpose(np.clip(data /2 + 0.5, 0, 1), (0,4,1,2,3))
print(base[0,0,0,0,:100], data[0,0,0,0,:100])
print(np.allclose(base,data, 1e-2, 1e-2))