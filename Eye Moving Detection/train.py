import torch
import torch.nn.functional as f
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
import os
from tqdm import tqdm


class EmdDataset(Dataset):
    def __init__(self, base_path: str = "images/", cuda=True):
        self.labels = os.listdir(base_path)
        self.one_hot_labels = f.one_hot(torch.arange(len(self.labels)))
        self.base_path = base_path
        self.len = sum(
            map(lambda x: len(os.listdir(os.path.join(base_path, x))), self.labels)
        )
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for label, one_hot_label in zip(self.labels, self.one_hot_labels):
            path = os.path.join(base_path, label)
            for file in os.listdir(path):
                file = os.path.join(path, file)
                image = read_image(file, mode=ImageReadMode.GRAY) / 255.0
                self.data.append(
                    (image, one_hot_label)
                    if not cuda
                    else (image.cuda(), one_hot_label.cuda())
                )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


BATCH_SIZE = 16
EPOCHS = 200


if __name__ == "__main__":
    dataset = EmdDataset(cuda=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = torch.nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 1),
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(128 * 5 * 4, 128),
        nn.Dropout(0.5, inplace=True),
        nn.LeakyReLU(),
        nn.Linear(128, len(dataset.labels)),
        nn.Softmax(dim=1),
    )
    # model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in tqdm(range(EPOCHS)):
        corrects = 0
        for image, label in dataloader:
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, torch.argmax(label, dim=1))
            loss.backward()
            optimizer.step()
            corrects += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(label, dim=1))
        if epoch and not epoch % 10:
            print("\nacc:", corrects / len(dataset))
            torch.save(model, "model.pth")
