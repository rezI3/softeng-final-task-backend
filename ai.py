from torch import nn

INPUT_SIZE = 60
OUTPUT_SIZE = 1000


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 500),
            nn.ReLU(),
            nn.Linear(500, OUTPUT_SIZE),

        )

    def forward(self, x):
        x = self.fc(x)
        return x