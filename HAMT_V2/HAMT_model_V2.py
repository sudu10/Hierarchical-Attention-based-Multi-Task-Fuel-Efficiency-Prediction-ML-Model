import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcitationBlock(nn.Module):

    def __init__(self, channels: int = 6, reduction_ratio: int = 2):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, C, T = x.shape
        u = self.squeeze(x).squeeze(-1)
        s = self.excitation(u)
        x_recalibrated = x * s.unsqueeze(-1)
        return x_recalibrated, s


class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels: int = 6, per_kernel_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(in_channels, per_kernel_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(per_kernel_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(in_channels, per_kernel_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(per_kernel_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_k9 = nn.Sequential(
            nn.Conv1d(in_channels, per_kernel_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(per_kernel_dim),
            nn.ReLU(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Conv1d(per_kernel_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.conv_k3(x)
        c2 = self.conv_k5(x)
        c3 = self.conv_k9(x)
        c_cat = torch.cat([c1, c2, c3], dim=1)
        P = self.projection(c_cat)
        return P.permute(0, 2, 1)


class BiLSTMEncoder(nn.Module):

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, P):
        H, _ = self.lstm(P)
        H = self.layer_norm(H)
        H = self.dropout(H)
        return H


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features):
        batch, C, _ = node_features.shape
        Wh = self.W(node_features)

        Wh_i = Wh.unsqueeze(2).expand(-1, -1, C, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, C, -1, -1)
        e_input = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leaky_relu(torch.matmul(e_input, self.a).squeeze(-1))

        attention = F.softmax(e, dim=-1)

        out = torch.bmm(attention, Wh)
        return F.elu(out), attention


class GraphAttentionNetwork(nn.Module):

    def __init__(self, input_dim: int = 128, graph_dim: int = 64,
                 num_channels: int = 6, num_heads: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads

        self.node_embed = nn.Linear(input_dim, graph_dim)

        head_dim = graph_dim // num_heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(graph_dim, head_dim)
            for _ in range(num_heads)
        ])

        self.readout = nn.Sequential(
            nn.Linear(graph_dim, input_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, H):
        batch, T, hidden = H.shape
        C = self.num_channels
        channel_dim = hidden // C

        if hidden % C == 0:
            H_channels = H.view(batch, T, C, channel_dim)
            node_inputs = H_channels.mean(dim=1)
        else:
            node_inputs = H.mean(dim=1).unsqueeze(1).expand(-1, C, -1)[:, :, :hidden // C + 1]
            node_inputs = node_inputs[:, :, :channel_dim] if node_inputs.shape[-1] > channel_dim else F.pad(
                node_inputs, (0, channel_dim - node_inputs.shape[-1]))

        nodes = self.node_embed(
            F.pad(node_inputs, (0, max(0, self.node_embed.in_features - node_inputs.shape[-1])))
            if node_inputs.shape[-1] < self.node_embed.in_features
            else node_inputs[:, :, :self.node_embed.in_features]
        )

        head_outputs = []
        all_attentions = []
        for head in self.attention_heads:
            head_out, attn = head(nodes)
            head_outputs.append(head_out)
            all_attentions.append(attn)

        n_prime = torch.cat(head_outputs, dim=-1)
        attention_weights = torch.stack(all_attentions, dim=1)

        g = self.readout(n_prime.mean(dim=1))

        return g, attention_weights


class CrossAttentionFusion(nn.Module):

    def __init__(self, hidden_dim: int = 128, vehicle_features: int = 7,
                 vehicle_emb_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vehicle_emb_dim = vehicle_emb_dim

        self.vehicle_embedding = nn.Sequential(
            nn.Linear(vehicle_features, vehicle_emb_dim),
            nn.BatchNorm1d(vehicle_emb_dim),
            nn.ReLU(inplace=True)
        )

        self.inject_graph = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inject_vehicle = nn.Linear(vehicle_emb_dim, hidden_dim, bias=False)

        self.kv_proj = nn.Linear(hidden_dim, vehicle_emb_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=vehicle_emb_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.layer_norm = nn.LayerNorm(vehicle_emb_dim)

        fusion_input_dim = vehicle_emb_dim + vehicle_emb_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, H, g, V):
        v_emb = self.vehicle_embedding(V)

        H_injected = H + self.inject_graph(g).unsqueeze(1) + self.inject_vehicle(v_emb).unsqueeze(1)

        H_proj = self.kv_proj(H)

        Q = self.kv_proj(H_injected)
        attn_output, cross_attn_weights = self.multihead_attn(Q, H_proj, H_proj)

        Z_final = self.layer_norm(Q + attn_output)

        Z_mean = Z_final.mean(dim=1)
        fused_input = torch.cat([Z_mean, v_emb, g], dim=1)
        u_final = self.fusion(fused_input)

        return u_final, v_emb, cross_attn_weights


class FuelLossHead(nn.Module):

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, u):
        return self.head(u)


class BehaviorClassificationHead(nn.Module):

    def __init__(self, input_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, u):
        return self.head(u)


class DriverProfileHead(nn.Module):

    def __init__(self, input_dim: int = 128, embedding_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(input_dim, 32)
        self.embedding = nn.Linear(32, embedding_dim)

    def forward(self, u):
        z = torch.tanh(self.fc(u))
        e = self.embedding(z)
        e_normalized = F.normalize(e, p=2, dim=1)
        return e_normalized


class RouteEfficiencyHead(nn.Module):

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, u):
        return self.head(u)


class HAMTFuelModelV2(nn.Module):

    def __init__(self, input_channels: int = 6, vehicle_features: int = 7,
                 num_behavior_classes: int = 6, hidden_dim: int = 128,
                 graph_dim: int = 64, num_gat_heads: int = 4,
                 num_cross_attn_heads: int = 4, se_reduction: int = 2):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        self.se_block = SqueezeExcitationBlock(input_channels, se_reduction)

        self.multi_scale_cnn = MultiScaleCNN(input_channels, per_kernel_dim=32, hidden_dim=hidden_dim)

        self.bilstm = BiLSTMEncoder(hidden_dim, hidden_dim, num_layers=2, dropout=0.3)

        self.gat = GraphAttentionNetwork(hidden_dim, graph_dim, input_channels, num_gat_heads)

        self.cross_attention = CrossAttentionFusion(
            hidden_dim, vehicle_features, vehicle_emb_dim=64, num_heads=num_cross_attn_heads
        )

        self.fuel_head = FuelLossHead(hidden_dim)
        self.behavior_head = BehaviorClassificationHead(hidden_dim, num_behavior_classes)
        self.driver_head = DriverProfileHead(hidden_dim, embedding_dim=16)
        self.route_head = RouteEfficiencyHead(hidden_dim)

    def forward(self, telemetry, vehicle_context):
        x_se, se_weights = self.se_block(telemetry)

        P = self.multi_scale_cnn(x_se)

        H = self.bilstm(P)

        g, gat_attention = self.gat(H)

        u_final, v_emb, cross_attn_weights = self.cross_attention(H, g, vehicle_context)

        fuel_loss = self.fuel_head(u_final)
        behavior_logits = self.behavior_head(u_final)
        driver_embedding = self.driver_head(u_final)
        route_efficiency = self.route_head(u_final)

        return {
            'fuel_loss': fuel_loss,
            'behavior_logits': behavior_logits,
            'driver_embedding': driver_embedding,
            'route_efficiency': route_efficiency,
            'se_weights': se_weights,
            'gat_attention': gat_attention,
            'cross_attention_weights': cross_attn_weights,
            'vehicle_embedding': v_emb,
            'graph_representation': g
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        module_params = {}
        for name, module in self.named_children():
            module_params[name] = sum(p.numel() for p in module.parameters())
        return {'total': total, 'trainable': trainable, 'per_module': module_params}


class MultiTaskLossV2(nn.Module):

    def __init__(self, alpha=0.45, beta=0.25, gamma=0.15, delta=0.10,
                 lambda1=1e-4, lambda2=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.fuel_loss_fn = nn.HuberLoss(delta=1.0)
        self.behavior_loss_fn = nn.CrossEntropyLoss()
        self.route_loss_fn = nn.MSELoss()

    def triplet_loss(self, anchor, positive, negative, margin=0.5):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def gat_entropy_regularizer(self, gat_attention):
        eps = 1e-8
        entropy = -(gat_attention * torch.log(gat_attention + eps)).sum(dim=-1).mean()
        return entropy

    def forward(self, outputs, targets, model=None):
        l_fuel = self.fuel_loss_fn(outputs['fuel_loss'], targets['fuel_loss'])
        l_behavior = self.behavior_loss_fn(outputs['behavior_logits'], targets['behavior_class'])
        l_route = self.route_loss_fn(outputs['route_efficiency'], targets['route_efficiency'])

        l_driver = torch.tensor(0.0, device=outputs['fuel_loss'].device)
        if 'driver_positive' in targets and 'driver_negative' in targets:
            l_driver = self.triplet_loss(
                outputs['driver_embedding'], targets['driver_positive'], targets['driver_negative']
            )

        total = (self.alpha * l_fuel + self.beta * l_behavior +
                 self.gamma * l_driver + self.delta * l_route)

        if model is not None:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            total = total + self.lambda1 * l2_reg

        if 'gat_attention' in outputs:
            gat_entropy = self.gat_entropy_regularizer(outputs['gat_attention'])
            total = total + self.lambda2 * gat_entropy

        return total, {
            'fuel': l_fuel.item(),
            'behavior': l_behavior.item(),
            'driver': l_driver.item() if isinstance(l_driver, torch.Tensor) else 0.0,
            'route': l_route.item(),
            'total': total.item()
        }


if __name__ == "__main__":
    model = HAMTFuelModelV2()
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable: {params['trainable']:,}")
    print("\nPer-module:")
    for name, count in params['per_module'].items():
        print(f"  {name}: {count:,}")

    batch = 4
    telemetry = torch.randn(batch, 6, 60)
    vehicle_ctx = torch.randn(batch, 7)
    outputs = model(telemetry, vehicle_ctx)
    print("\nOutput shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")