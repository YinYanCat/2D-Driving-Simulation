import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, image_w, image_h):
        super().__init__()

        # CNN para imagen
        out1_cnn = 16
        out2_cnn = 32
        out3_cnn = 64
        self.conv1 = nn.Conv2d(3, out1_cnn,kernel_size=8, stride=4, dilation=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out1_cnn)
        self.conv2 = nn.Conv2d(out1_cnn, out2_cnn,kernel_size=4, stride=2, dilation=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out2_cnn)
        self.conv3 = nn.Conv2d(out2_cnn, out3_cnn,kernel_size=3, stride=1, dilation=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out3_cnn)

        self.cnn_output_dim = out3_cnn * 4 * 4

        #  Multilayer Perceptron para inputs continuos
        out1_mlp = 128
        out2_mlp = 256
        self.fc_state1 = nn.Linear(state_dim, out1_mlp)
        self.fc_state2 = nn.Linear(out1_mlp,out2_mlp)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # MLP final para cnn y estado
        combined_dim = self.cnn_output_dim + out2_mlp
        out1_comb = 512
        out2_comb = 256
        self.fc1 = nn.Linear(combined_dim, out1_comb)
        self.fc2 = nn.Linear(out1_comb, out2_comb)

        # Salidas por acción
        self.action_out = nn.Linear(out2_comb, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # inicialización de pesos con Kaiming permite diversificar los resultados iniciales de las neuronas, para eventualmente especializase
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, img, state):

        # imagen
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # pooling
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # variables continuas
        s = F.relu(self.fc_state1(state))
        s = F.relu(self.fc_state2(s))
        s = self.dropout(s)

        combined = torch.cat([x, s], dim=1)

        h = F.relu(self.fc1(combined))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        action_q = self.action_out(h)

        return action_q
