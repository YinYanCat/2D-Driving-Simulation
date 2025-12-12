import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, image_w, image_h):
        super().__init__()

        # CNN para imagen
        out1_cnn = 32
        out2_cnn = 64
        out3_cnn = 128
        out4_cnn = 128

        self.conv1 = nn.Conv2d(3, out1_cnn, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(out1_cnn)
        self.conv2 = nn.Conv2d(out1_cnn, out2_cnn ,kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(out2_cnn)
        self.conv3 = nn.Conv2d(out2_cnn, out3_cnn, kernel_size=3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(out3_cnn)
        self.conv4 = nn.Conv2d(out3_cnn, out4_cnn, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out4_cnn)

        self.cnn_output_dim = out4_cnn * 4 * 4

        #  Multilayer Perceptron para inputs continuos
        out1_mlp = 128
        out2_mlp = 256
        out3_mlp = 256
        self.fc_state1 = nn.Linear(state_dim, out1_mlp)
        self.ln_state1 = nn.LayerNorm(out1_mlp)
        self.fc_state2 = nn.Linear(out1_mlp,out2_mlp)
        self.ln_state2 = nn.LayerNorm(out2_mlp)
        self.fc_state3 = nn.Linear(out2_mlp,out3_mlp)
        self.ln_state3 = nn.LayerNorm(out3_mlp)

        # MLP final para cnn y estado
        combined_dim = self.cnn_output_dim + out3_mlp

        out1_comb = 512
        out2_comb = 512
        out3_comb = 256

        self.fc_combined1 = nn.Linear(combined_dim, out1_comb)
        self.ln_combined1 = nn.LayerNorm(out1_comb)
        self.fc_combined2 = nn.Linear(out1_comb, out2_comb)
        self.ln_combined2 = nn.LayerNorm(out2_comb)
        self.fc_combined3 = nn.Linear(out2_comb, out3_comb)
        self.ln_combined3 = nn.LayerNorm(out3_comb)
        # Salidas por acción
        self.action_out = nn.Linear(out3_comb, action_dim)

        # Dropout
        self.dropout = nn.Dropout(0.3)

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
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # pooling
        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # variables continuas
        s = F.relu(self.ln_state1(self.fc_state1(state)))
        s = self.dropout(s)

        s = F.relu(self.ln_state2(self.fc_state2(s)))
        s = self.dropout(s)

        s = F.relu(self.ln_state3(self.fc_state3(s)))
        s = self.dropout(s)


        combined = torch.cat([x, s], dim=1)

        h = F.relu(self.ln_combined1(self.fc_combined1(combined)))
        h = self.dropout(h)

        h = F.relu(self.ln_combined2(self.fc_combined2(h)))
        h = self.dropout(h)

        h = F.relu(self.ln_combined3(self.fc_combined3(h)))

        action_q = self.action_out(h)

        return action_q
