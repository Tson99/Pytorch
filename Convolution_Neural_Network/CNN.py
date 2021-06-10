import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import resnet50
from torchvision import transforms


# def print_info(self, input, output):
#     # input là 1 tuple các input
#     # output là 1 tensor, giá trị của output năm ở output.data
#     print('Inside ' + self.__class__.__name__ + ' forward')
#
#     print('')
#     print('input: ', type(input), ', len: ', len(input))
#     print('input[0]: ', type(input[0]), ', shape: ', input[0].shape)
#     print('output: ', type(output), ', len: ', len(output), output.data.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ConvLayer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # FClayer
        self.fc1 = nn.Linear(in_features=16 * 16 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(input=x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(input=self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            if torch.cuda.is_available():  # Check co GPU khong?
                imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        correct = 0
        # Tính độ chính xác trên tập validation
        with torch.no_grad():
            for images, labels in val_loader:
                if torch.cuda.is_available():  # Check co GPU khong?
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                correct += c.sum()
        if epoch == 1 or epoch % 1 == 0:
            print('Epoch {}, Training loss {}, Val accuracy {}'.format(
                epoch,
                loss_train / len(train_loader),
                correct / len(val_loader))
            )


if __name__ == '__main__':
    # Transforms
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4915, 0.4823, 0.4468),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # Load datasets
    data_path = '../models'
    cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms)
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=True)

    # Training
    model = Net()
    print(len(list(model.parameters())))
    if torch.cuda.is_available():
        model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    training_loop(
        n_epochs=100,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader
    )
