import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler


def get_image_size(image_size, kernel_sizes, stride_sizes=(1, 1, 1), padding_sizes=(2, 2, 2), pooling_sizes=(2, 2, 2)):
    for index in range(len(kernel_sizes)):
        kernel_size = kernel_sizes[index]
        stride_size = stride_sizes[index]
        padding_size = padding_sizes[index]
        image_size = math.floor((image_size - kernel_size + 2 *
                                 padding_size) / stride_size) + 1
        image_size /= pooling_sizes[index]
    return int(image_size)


class CNN(nn.Module):
    def __init__(self, kernel_sizes, activation_function):
        super().__init__()
        num_classes = 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=kernel_sizes[0], stride=1, padding=2),
            activation_function(),
            nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=kernel_sizes[1], stride=1, padding=2),
            activation_function(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=kernel_sizes[2], stride=1, padding=2),
            activation_function(),
            nn.MaxPool2d(2, 2))
        image_size = get_image_size(256, kernel_sizes)
        self.fc = nn.Linear(
            in_features=image_size * image_size * 32,
            out_features=num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class COVIDModel:
    def __init__(self, activation_function='ReLU', kernel_sizes=(7, 5, 3), regularization_strategy=None):
        self.kernel_sizes = kernel_sizes
        self.activation_function = getattr(nn, activation_function)
        self.regularization_strategy = regularization_strategy

    def load_images(self, images_path, seed=10):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        images_dataset = datasets.ImageFolder(
            images_path, transform=transform)
        train_dataset, rest_dataset = random_split(
            images_dataset, [2000, 541], generator=torch.Generator().manual_seed(seed))
        eval_dataset, test_dataset = random_split(
            rest_dataset, [241, 300], generator=torch.Generator().manual_seed(seed))
        self.train_loader = DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=32)
        self.eval_loader = DataLoader(dataset=eval_dataset,
                                      shuffle=False, batch_size=32)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      shuffle=False, batch_size=32)

    def train(self, num_epochs=5, learning_rate=0.001):
        model = CNN(kernel_sizes=self.kernel_sizes,
                    activation_function=self.activation_function)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(self.train_loader)
        list_loss = []
        index = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # images = images.to(device)
                # labels = labels.to(device)

                output = model(images)
                loss = loss_fn(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                list_loss.append(
                    {
                        "metrics": {"loss": loss.item()}, 'step': index
                    }
                )
                index += 1

                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            print('Finished Training')
            return list_loss
