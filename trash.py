import torch
from numpy import inf
import torch.nn as nn

"""rand = torch.randn(16, 100)
print(rand.shape)
lab = torch.randint(0, 10, (16,))
print(lab.shape, lab)
emb = nn.Embedding(10, 10)
lab_emb = emb(lab)
print(lab_emb.shape)
conct = torch.cat((lab_emb, rand), -1)
print(conct.shape)"""

print(-inf + inf)
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
torch.backends.cudnn.enabled = True
print(torch.backends.cudnn.enabled)

for i in range(1, 10):
    print(i)