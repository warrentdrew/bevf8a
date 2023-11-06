import numpy as np

pd = np.load("pd.npy")
pth = np.load("/mnt/zhuyipin/idg/lidarrcnn/BEVFusion/pth.npy")

print("pd shape: ", pd.shape)
print("pth shape: ", pth.shape)

np.testing.assert_allclose(pd, pth)
