import torch.nn as nn
from pathlib import Path
import torch
import torch.nn.functional as F
from constants import HIDDEN_DIMS, NUM_CLASSES, CHECKPOINT_PATH
 
class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
 
    def forward(self, x):
        return self.act(self.ff(x))
 
 
class EqualityMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, num_classes=NUM_CLASSES):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.blocks = nn.ModuleList(
            Block(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        self.score = nn.Linear(dims[-1], num_classes)
 
    def forward(self, inputs_embeds):
        h = inputs_embeds if inputs_embeds.dim() == 3 else inputs_embeds.unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.score(h).squeeze(1)   # [N, num_classes]

 
 
@torch.no_grad()
def exact_accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(-1) == y).float().mean().item()
 
 
def train_backbone(seed=0, epochs=1000, lr=2e-3, batch_size=256, device="cpu"):
    ev = build_entity_vectors()
    Xtr, ytr = make_dataset(ev, FACTUAL_TRAIN_SIZE, seed)
    Xva, yva = make_dataset(ev, FACTUAL_VAL_SIZE, seed + 1)
    Xtr, ytr, Xva, yva = (t.to(device) for t in (Xtr, ytr, Xva, yva))
 
    torch.manual_seed(seed)
    model = EqualityMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
 
    cnt = 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), device=device).split(batch_size):
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        acc = exact_accuracy(model, Xva, yva)
        print(f"epoch {epoch + 1:2d}  val_exact={acc:.4f}")
        cnt = cnt + 1 if acc >= 1.0 else 0
        if cnt == 8:
            break
    return model, ev
 
 
def save_backbone(model, path=CHECKPOINT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
 
 
def load_backbone(path=CHECKPOINT_PATH, device="cpu"):
    model = EqualityMLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
 
 
if __name__ == "__main__":
    model, _ = train_backbone()
    save_backbone(model)
    print(f"saved -> {CHECKPOINT_PATH}")