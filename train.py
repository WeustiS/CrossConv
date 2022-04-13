import torch
from torch import nn
import torchvision
from torchvision.transforms import transforms
from CoAtNet import CoAtNet
from CrossConv import CrossConv
from CrossConvPlusConvMixer import CrossConvWMix
from utils import seed_everything, acc, accuracy
import wandb
init = True
from tqdm import tqdm



seed_everything(42)

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = torchvision.datasets.ImageFolder(r"/raid/projects/weustis/data/imagenet_subset/train", transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(r"/raid/projects/weustis/data/imagenet_subset/val", transform=test_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
model = CoAtNet((224, 224), 3, num_blocks, channels, num_classes=200).cuda()

crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=.001, weight_decay=.03, momentum=.9)

for epoch in range(30):
    # train loop
    model.train()
    train_loss_total = 0
    train_top1_total = 0
    train_top5_total = 0
    
    test_loss_total = 0
    test_top1_total = 0
    test_top5_total = 0
    
    for x,y in tqdm(train_dataloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        loss = crit(pred, y)
        train_loss_total += loss.item()/len(x)

        top1, top5 = accuracy(pred, y)
        train_top1_total += top1.item() * len(x)
        train_top5_total += top5.item() * len(x)
        
        loss.backward()
        opt.step()
       
    model.eval()
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = crit(pred, y)
            test_loss_total += loss.item()/len(x)
            top1, top5 = accuracy(pred, y)
            test_top1_total += top1.item() * len(x)
            test_top5_total += top5.item() * len(x)
    if init:
        wandb.init(project="crossconv")
        init = False
    wandb.log({
        "train_loss": train_loss_total,
        "train_top1": train_top1_total/len(train_dataset),
        "train_top5": train_top5_total/len(train_dataset),
        "test_loss": test_loss_total,
        "test_top1": test_top1_total/len(test_dataset),
        "test_top5": test_top5_total/len(test_dataset)
    })
       