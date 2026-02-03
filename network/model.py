import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out

class OnitamaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        self.name = model_config['name']
        self.num_res_blocks = model_config.get('num_res_blocks', 4)
        self.num_channels = model_config.get('num_channels', 64)
        self.input_planes = model_config.get('input_planes', 9)
        self.board_size = model_config.get('board_size', 5)
        self.action_size = model_config.get('action_space_size', 1252)
        self.pol_channels = model_config.get('policy_head_channels', 16)
        self.val_channels = model_config.get('value_head_channels', 32)
        self.val_hidden = model_config.get('value_head_hidden', 128)
        self.wdl = model_config.get('wdl', False)
        self.flatten = nn.Flatten()
        
        self.conv_input = nn.Conv2d(self.input_planes, self.num_channels, kernel_size=1)
        
        self.res_tower = nn.Sequential(
            *[ResBlock(self.num_channels) for _ in range(self.num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(self.num_channels, self.pol_channels, kernel_size=1) 
        self.policy_fc = nn.Linear(self.pol_channels * self.board_size * self.board_size, self.action_size)

        self.value_conv = nn.Conv2d(self.num_channels, self.val_channels, kernel_size=1) 
        self.value_fc1 = nn.Linear(self.val_channels * self.board_size * self.board_size, self.val_hidden)

        if self.wdl:
            self.value_fc2 = nn.Linear(self.val_hidden, 3) 
        else:
            self.value_fc2 = nn.Linear(self.val_hidden, 1) 

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_tower(x)

        p = F.relu(self.policy_conv(x))
        p = self.flatten(p) 
        p = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = self.flatten(v)
        v = F.relu(self.value_fc1(v))
        if self.wdl:
            v = torch.tanh(self.value_fc2(v)) 
        else:
            v = self.value_fc2(v)

        return p, v

