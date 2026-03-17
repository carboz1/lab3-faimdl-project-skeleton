from torch import nn

import torch
import checkpoints.model

model = checkpoints.model.CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)