import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # CNN para imagen
        out1_cnn = ...
        out2_cnn = ...
        out3_cnn = ...
        self.conv1 = nn.Conv2d(3, out1_cnn,kernel_size=..., stride=...)
        self.conv2 = nn.Conv2d(out1_cnn, out2_cnn)
        self.conv3 = nn.Conv2d(out2_cnn, out3_cnn)

        self.cnn_output_dim = ... # pixeles de la imagen

        #  Multilayer Perceptron para inputs continuos
        self.fc_state1 = nn.Linear(state_dim, ...)
        self.fc_state2 = nn.Linear()

        # MLP final para cnn y estado
        self.fc1 = nn.Linear()
        self.out = nn.Linear()

    def foward(self, img, state):

        # imagen
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # variables continuas
        s = F.relu(self.fc_state1(state))
        s = F.relu(self.fc_state2(s))

        combined = torch.cat([x, s], dim=1)

        f = F.relu(self.fc1(combined))
        q_values = self.out(h)

        return q_values