import torch
from torch import nn


def nce_loss(input1,input2,device,temperature=0.5):
    cos = nn.CosineSimilarity(dim=-1).to(device)
    sim11 = cos(input1.unsqueeze(-2), input1.unsqueeze(-3)) / temperature
    sim22 = cos(input2.unsqueeze(-2), input2.unsqueeze(-3)) / temperature
    sim12 = cos(input1.unsqueeze(-2), input2.unsqueeze(-3)) / temperature

    d = sim12.shape[-1]

    sim11[..., range(d), range(d)] = float('-inf')
    sim22[..., range(d), range(d)] = float('-inf')
    raw_scores1 = torch.cat([sim12, sim11], dim=-1)
    raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
    logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
    criterion = nn.CrossEntropyLoss().to(device)
    nce_loss = criterion(logits, labels)
    return nce_loss