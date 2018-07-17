import torch
import torch.nn as nn


class Siamese(nn.Module):
    def __init__(self, dropout=False, normalization=True):
        super(Siamese, self).__init__()

        if normalization:
            # left branch: vocal imitations
            self.left_branch = nn.Sequential(
                nn.Conv2d(1, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
            )

            # right branch: reference sounds
            self.right_branch = nn.Sequential(
                nn.Conv2d(1, 24, (5, 5)),
                nn.BatchNorm2d(24, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),
                nn.Conv2d(24, 48, (5, 5)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),
                nn.Conv2d(48, 48, (5, 5)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
            )
        else:
            # left branch: vocal imitations
            self.left_branch = nn.Sequential(
                nn.Conv2d(1, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
            )

            # right branch: reference sounds
            self.right_branch = nn.Sequential(
                nn.Conv2d(1, 24, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),
                nn.Conv2d(24, 48, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),
                nn.Conv2d(48, 48, (5, 5)),
                nn.ReLU(),
            )

        if dropout:
            self.fully_connected = nn.Sequential(
                nn.Dropout(p=.2),
                nn.Linear(48 * 55 + 48 * 25 * 2, 108),
                nn.ReLU(),
                nn.Dropout(p=.2),
                nn.Linear(108, 1),
                nn.Sigmoid()
            )
        else:
            self.fully_connected = nn.Sequential(
                nn.Linear(48 * 55 + 48 * 25 * 2, 108),
                nn.ReLU(),
                nn.Linear(108, 1),
                nn.Sigmoid()
            )

    def forward(self, left, right):
        # raise RuntimeError
        # Calculate both CNN branches
        print left.shape
        left_output = self.left_branch(left)
        right_output = self.right_branch(right)

        # Flatten and concatenate them
        left_reshaped = left_output.view(len(left_output), -1)
        right_reshaped = right_output.view(len(right_output), -1)
        # noinspection PyUnresolvedReferences
        concatenated = torch.cat((left_reshaped, right_reshaped), dim=1)

        # Calculate the FCN and flatten it
        output = self.fully_connected(concatenated)
        return output.view(-1)