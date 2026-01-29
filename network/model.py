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
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OnitamaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.name = config['model']['name']
        self.num_res_blocks = config['model']['num_res_blocks']
        self.num_channels = config['model']['num_channels']
        self.input_planes = config['model']['input_planes']
        self.board_size = config['model']['board_size']
        self.action_size = config['model']['action_space_size']
        self.pol_hidden = config['model']['policy_head_hidden']
        self.val_hidden = config['model']['value_head_hidden']

        
        self.conv_input = nn.Sequential(
            nn.Conv2d(self.input_planes, self.num_channels, kernel_size=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU()
        )
        
        self.res_tower = nn.Sequential(
            *[ResBlock(self.num_channels) for _ in range(self.num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(self.num_channels, self.pol_hidden, kernel_size=1) 
        self.policy_bn = nn.BatchNorm2d(self.pol_hidden)
        self.policy_fc = nn.Linear(self.pol_hidden * self.board_size * self.board_size, self.action_size)

        self.value_conv = nn.Conv2d(self.num_channels, 1, kernel_size=1) 
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_size * self.board_size, self.val_hidden)
        self.value_fc2 = nn.Linear(self.val_hidden, 1) 

    def forward(self, x):
        # 1. Backbone
        x = self.conv_input(x)
        x = self.res_tower(x)

        # 2. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) 
        p = self.policy_fc(p)
        log_policy = F.log_softmax(p, dim=1) 

        # 3. Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) 
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) 

        return log_policy, v


def load_model_from_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return OnitamaNet(config)
