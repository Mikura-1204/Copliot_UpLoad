#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - 该网络为复现的博士论文中提到的生成对抗网络结构的增强版本
# - 信号样本被形变为16*16，fc为全连接层，LR代表Leaky ReLU，
# - conv表示卷积，deconv表示反卷积。
# - 在Leaky ReLU中设置a=0.2, dropout的参数设为0.4
# - 第二版增强版本增加了更多的卷积和反卷积层以提升特征建模能力
# - 第三版优化版本解决了判别器过强和训练不稳定的问题
#####------------------------------------------------------------------------------------#####

import torch
import torch.nn as nn
import torch.nn.functional as F

class My_Generator_2(nn.Module):
    def __init__(self, latent_dim=100):
        super(My_Generator_2, self).__init__()
        
        # 保存latent_dim
        self.latent_dim = latent_dim
        
        # 全连接层将噪声向量转换为2048维
        self.fc = nn.Linear(latent_dim, 2048)
        self.batch_norm1d = nn.BatchNorm1d(2048)  # 增加批归一化
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 增强版: 增加中间层数，使用ResNet风格的跳跃连接
        # 开始部分: 4×4×128
        self.deconv_initial = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 增强版反卷积层1: 4×4 -> 8×8
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 投影层: 从192通道降为128通道
        self.projection1 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        
        # 增强版反卷积层2: 8×8 -> 16×16
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 160, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 投影层: 从160通道降为128通道
        self.projection2 = nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0)
        
        # 特征提炼层 - 回退到简化版本，避免过度复杂化
        # 用于进一步提取特征
        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),  # 1x1卷积减轻过拟合
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 最终卷积层，生成2通道输出（用于IQ两个通道）
        self.conv_out = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)  # 16×16×128 -> 16×16×2
        self.tanh = nn.Tanh()  # 输出范围限制在[-1, 1]
        
    def forward(self, x):
        # x的输入形状: [batch_size, latent_dim]
        batch_size = x.size(0)
        
        # 减少噪声添加，保持稳定性
        if self.training:
            noise_scale = 0.01  # 大幅减小噪声强度
            x = x + torch.randn_like(x) * noise_scale
        
        # 全连接层将噪声映射到2048维
        x = self.fc(x)
        x = self.batch_norm1d(x)
        x = self.leaky_relu(x)
        
        # 重塑为4×4×128的特征图 (2048 = 4×4×128)
        x = x.view(batch_size, 128, 4, 4)
        x = self.deconv_initial(x)
        
        # 增强版反卷积层1
        x1 = self.deconv1(x)  # 8×8×192
        x1 = self.projection1(x1)  # 8×8×128
        
        # 增强版反卷积层2
        x2 = self.deconv2(x1)  # 16×16×160
        x2 = self.projection2(x2)  # 16×16×128
        
        # 特征提炼层 (残差连接)
        x_refine = self.refine(x2)
        x = x2 + x_refine * 0.5  # 减少残差连接的影响，避免特征过于复杂
        
        # 最终卷积层生成输出
        x = self.conv_out(x)  # 16×16×2
        x = self.tanh(x)
        
        # 将16×16×2的输出重塑为信号数据格式 [batch_size, 2, 128]
        x_reshaped = x.view(batch_size, 2, 16*16)
        x_output = x_reshaped[:, :, :128]
        
        # 检查输出形状
        assert x_output.size() == (batch_size, 2, 128), f"输出形状错误: {x_output.size()}, 预期: [{batch_size}, 2, 128]"
        
        return x_output
    
class My_Discriminator_2(nn.Module):
    def __init__(self, num_classes=11):
        super(My_Discriminator_2, self).__init__()
        
        # 调整判别器参数，防止过强
        self.label_smoothing = 0.1  # 标签平滑因子
        self.instance_noise = 0.05  # 适中的实例噪声强度
        
        # 首先将[batch_size, 2, 128]的信号直接重塑为16×16×1的图像格式
        self.reshape_signal = lambda x: self._reshape_to_image(x)
        
        # 回退到原来的结构，避免层级过深造成过拟合
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),  # 16×16×1 -> 8×8×96
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 128, kernel_size=4, stride=2, padding=1),  # 8×8×96 -> 4×4×128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 192, kernel_size=4, stride=2, padding=1),  # 4×4×128 -> 2×2×192
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # 简化注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 将最终特征图重塑为向量 (2×2×192=768)
        self.reshape_features = lambda x: x.view(x.size(0), -1)
        
        # 减少Dropout概率，防止信息损失过多
        self.dropout = nn.Dropout(p=0.3)
        
        # 特征融合层
        self.fc_fusion = nn.Linear(768, 384)
        self.batch_norm = nn.BatchNorm1d(384)  # 添加批归一化
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 分类输出
        self.fc_real_fake = nn.Linear(384, 2)  # 真/假二分类
        self.fc_classes = nn.Linear(384, num_classes)  # 调制类型分类
        
    def _reshape_to_image(self, x):
        # 将[batch_size, 2, 128]的信号直接重塑为[batch_size, 1, 16, 16]的图像格式
        batch_size = x.size(0)
        
        # 检查输入格式
        if x.size(1) != 2 or x.size(2) != 128:
            raise ValueError(f"输入数据形状应为[batch_size, 2, 128]，但得到{x.size()}")
        
        # 将两个通道的128点信号合并为一个256点的序列
        x_combined = torch.cat([x[:, 0, :], x[:, 1, :]], dim=1)  # [batch_size, 256]
        
        # 重塑为16×16，增加通道维度
        x_reshaped = x_combined.view(batch_size, 1, 16, 16)
        
        return x_reshaped
    
    def apply_instance_noise(self, x, noise_scale):
        # 实例噪声：添加高斯噪声到输入
        if self.training and noise_scale > 0:
            return x + torch.randn_like(x) * noise_scale
        return x
        
    def forward(self, x, training_step=None):
        # x的输入形状: [batch_size, 2, 128]
        x = self.reshape_signal(x)  # 变为[batch_size, 1, 16, 16]
        
        # 动态噪声衰减：根据训练步骤减少噪声
        noise_scale = 0
        if self.training and training_step is not None:
            # 从initial_noise_scale线性衰减到0，在前5000步
            max_steps = 5000  # 缩短噪声衰减时间
            step = min(training_step, max_steps)
            noise_scale = self.instance_noise * (1 - step / max_steps)
        
        # 添加实例噪声（训练初期会更强）
        x = self.apply_instance_noise(x, noise_scale)
        
        # 层级特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x_feat = self.conv3(x)  # 用于特征匹配
        
        # 应用注意力机制
        attention_map = self.attention(x_feat)
        x_feat_attended = x_feat * (0.5 + 0.5 * attention_map)  # 平衡注意力机制的影响
        
        # reshape得到向量
        x_reshaped = self.reshape_features(x_feat_attended)
        x_drop = self.dropout(x_reshaped)
        
        # 特征融合
        x_fused = self.fc_fusion(x_drop)
        x_fused = self.batch_norm(x_fused)  # 添加批归一化
        x_fused = self.relu(x_fused)
        x_fused = self.dropout(x_fused)
        
        # 输出真伪判断和类别判断
        real_fake_logits = self.fc_real_fake(x_fused)
        class_logits = self.fc_classes(x_fused)
        
        return real_fake_logits, class_logits

    def extract_features(self, x):
        # 特征提取函数，用于特征匹配损失
        x = self.reshape_signal(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_feat = self.conv3(x)
        
        # 不应用注意力机制 (保持更纯净的特征)
        return x_feat

    def extract_multilevel_features(self, x):
        """提取多层次特征，返回中间层和最终层特征，用于更高级的特征匹配"""
        x = self.reshape_signal(x)
        x = self.conv1(x)
        x_mid = self.conv2(x)
        x_feat = self.conv3(x_mid)
        
        # 保持更纯净的特征表示
        return x_mid, x_feat