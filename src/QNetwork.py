import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class QNetwork(nn.Module):
    def __init__(self, state_dim, gear_dim, brake_dim, accel_dim, steer_dim, image_w, image_h):
        super().__init__()

        # CNN para imagen
        out1_cnn = 8
        out2_cnn = 16
        out3_cnn = 32
        self.conv1 = nn.Conv2d(3, out1_cnn,kernel_size=4, stride=2, dilation=1, padding=0)
        self.conv2 = nn.Conv2d(out1_cnn, out2_cnn,kernel_size=2, stride=2, dilation=1, padding=0)
        self.conv3 = nn.Conv2d(out2_cnn, out3_cnn,kernel_size=1, stride=1, dilation=1, padding=0)

        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_w, 4, 2), 2, 2), 1, 1)
        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_h, 4, 2), 2, 2), 1, 1)
        self.cnn_output_dim = out3_cnn * w_out * h_out

        #  Multilayer Perceptron para inputs continuos
        self.fc_state1 = nn.Linear(state_dim, 128)
        self.fc_state2 = nn.Linear(128,128)

        # MLP final para cnn y estado
        self.fc1 = nn.Linear(self.cnn_output_dim + 128, 512)

        # Salidas por acción
        self.gear_out = nn.Linear(512, gear_dim)
        self.brake_out = nn.Linear(512, brake_dim)
        self.accel_out = nn.Linear(512, accel_dim)
        self.steer_out = nn.Linear(512, steer_dim)


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

        h = F.relu(self.fc1(combined))

        gear_q = self.gear_out(h)
        brake_q = self.brake_out(h)
        accel_q = self.accel_out(h)
        steer_q = self.steer_out(h)

        return gear_q, brake_q, accel_q, steer_q
