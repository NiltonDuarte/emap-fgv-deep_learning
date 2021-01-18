import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.metrics import confusion_matrix


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
    def __init__(self, kernel_sizes, activation_function, regularization_strategy='without'):
        super().__init__()
        num_classes = 2
        if regularization_strategy == 'without':
            regularization_strategy_layer = [
                nn.Identity(),
                nn.Identity(),
                nn.Identity()]
        elif regularization_strategy == 'Dropout':
            regularization_strategy_layer = [
                nn.Dropout(0.2),
                nn.Dropout(0.2),
                nn.Dropout(0.2)]
        elif regularization_strategy == 'BatchNorm2d':
            regularization_strategy_layer = [
                nn.BatchNorm2d(16),
                nn.BatchNorm2d(32),
                nn.BatchNorm2d(32)]
        else:
            raise NotImplementedError(
                f'regularization_strategy {regularization_strategy} not implemented')
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=kernel_sizes[0], stride=1, padding=2),
            regularization_strategy_layer[0],
            activation_function(),
            nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=kernel_sizes[1], stride=1, padding=2),
            regularization_strategy_layer[1],
            activation_function(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=kernel_sizes[2], stride=1, padding=2),
            regularization_strategy_layer[2],
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
        self._confusion_matrix = None
        self.kernel_sizes = kernel_sizes
        self.activation_function = getattr(nn, activation_function)
        self.regularization_strategy = regularization_strategy
        self.model = CNN(kernel_sizes=self.kernel_sizes,
                         activation_function=self.activation_function,
                         regularization_strategy=regularization_strategy)

    def load_images(self, images_path, seed=10):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        images_dataset = datasets.ImageFolder(
            images_path, transform=transform)
        train_dataset, eval_dataset, test_dataset = random_split(
            images_dataset, [2000, 241, 300], generator=torch.Generator().manual_seed(seed))
        self.train_loader = DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=32)
        self.eval_loader = DataLoader(dataset=eval_dataset,
                                      shuffle=False, batch_size=32)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      shuffle=False, batch_size=32)

    def train(self, num_epochs=5, learning_rate=0.001):
        model = self.model
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

        #print('warning: Finished Training at first epoch')
        return list_loss

    def get_metrics(self, data_loader):
        model = self.model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            # correct = 0
            # total = 0
            self._confusion_matrix = np.zeros((2, 2))
            for images, labels in data_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                partial_result = confusion_matrix(
                    predicted.view(-1), labels.view(-1))
                self._confusion_matrix += partial_result

        metrics = {}
        metrics["accuracy"] = self.accuracy()
        metrics["precision"] = self.precision()
        metrics["true_positive_rate"] = self.true_positive_rate()
        metrics["true_negative_rate"] = self.true_negative_rate()
        metrics["mcc"] = self.mcc()
        return {"metrics": metrics}

    def get_eval_metrics(self):
        return self.get_metrics(self.eval_loader)

    def get_test_metrics(self):
        return self.get_metrics(self.test_loader)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.ravel()

    def true_negative(self):
        return self.confusion_matrix[0]

    def false_positive(self):
        return self.confusion_matrix[1]

    def false_negative(self):
        return self.confusion_matrix[2]

    def true_positive(self):
        return self.confusion_matrix[3]

    def precision(self):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives
        and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label
        as positive a sample that is negative.
        """
        return self.true_positive() / (self.true_positive() + self.false_positive())

    def true_positive_rate(self):
        """
        The recall (true_positive_rate) is the ratio tp / (tp + fn) where tp is the number
        of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        """
        return self.true_positive() / (self.true_positive() + self.false_negative())

    def true_negative_rate(self):
        """
        Specificity (true_negative_rate) measures the proportion of negatives that are correctly identified
        tn / (tn + fp)
        """
        return self.true_negative() / (self.true_negative() + self.false_positive())

    def accuracy(self):
        return (self.true_positive() + self.true_negative()) / sum(self.confusion_matrix)

    def mcc(self):
        tp = self.true_positive()
        tn = self. true_negative()
        fp = self.false_positive()
        fn = self.false_negative()
        num = (tp * tn) - (fp * fn)
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        denom = math.sqrt(denom)
        mcc = num / denom
        return mcc
