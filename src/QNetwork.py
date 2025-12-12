import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, image_w, image_h):
        super().__init__()

        # CNN para imagen
        out1_cnn = 32
        out2_cnn = 64
        out3_cnn = 64

        self.conv1 = nn.Conv2d(3, out1_cnn, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(out1_cnn, out2_cnn ,kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(out2_cnn, out3_cnn, kernel_size=3, stride=1, padding=0)

        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_w, 8, 4), 4, 2), 3, 1)
        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_w, 8, 4), 4, 2), 3, 1)
        self.cnn_output_dim = out3_cnn * w_out * h_out

        #  Multilayer Perceptron para inputs continuos
        out1_mlp = 128
        out2_mlp = 128

        self.fc_state1 = nn.Linear(state_dim, out1_mlp)
        self.fc_state2 = nn.Linear(out1_mlp,out2_mlp)

        # MLP final para cnn y estado
        combined_dim = self.cnn_output_dim + out2_mlp

        out1_comb = 256
        out2_comb = 128

        self.fc_combined1 = nn.Linear(combined_dim, out1_comb)
        self.fc_combined2 = nn.Linear(out1_comb, out2_comb)

        # Salidas por acción
        self.action_out = nn.Linear(out2_comb, action_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        # inicialización de pesos con Kaiming, ayuda a evitar vanishing/exploding gradients, permite diversificar los resultados iniciales de las neuronas, para eventualmente especializase
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, img, state):

        # imagen
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # variables continuas
        s = F.relu(self.fc_state1(state))
        s = F.relu(self.fc_state2(s))

        combined = torch.cat([x, s], dim=1)
        h = F.relu(self.fc_combined1(combined))
        h = self.dropout(h)
        h = F.relu(self.fc_combined2(h))

        action_q = self.action_out(h)

        return action_q
