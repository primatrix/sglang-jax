import json
import numpy as np

with open("base.json","r") as f:
    base = np.array(json.loads(f.read())["after"]["output"])

with open("result.json","w") as f:
    data = np.array(json.loads(f.read())["data"])

print(np.allclose(base,data, 1e-5, 1e-5))