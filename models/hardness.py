import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningNorm(nn.Module):
    """
    在线归一化模块 (Running Normalization)。
    用于将不同量纲的难度指标（Loss, Cosine等）统一标准化到 N(0,1)。
    """
    def __init__(self, momentum=0.02, eps=1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("inited", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.numel() == 0: return
        m = x.mean()
        v = x.var(unbiased=False) + self.eps
        if self.inited.item() == 0:
            self.mean.copy_(m)
            self.var.copy_(v)
            self.inited.fill_(1)
        else:
            self.mean.mul_(1 - self.momentum).add_(self.momentum * m)
            self.var.mul_(1 - self.momentum).add_(self.momentum * v)

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

class HardnessBank:
    """
    难度记忆库，利用 EMA 平滑历史难度，减少 Batch 采样噪声。
    """
    def __init__(self, num_samples: int, momentum: float = 0.05, device: str = "cpu"):
        self.num_samples = num_samples
        self.momentum = momentum
        self.device = device
        self.h = torch.zeros(num_samples, device=device)
        self.cnt = torch.zeros(num_samples, device=device)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, h_new: torch.Tensor):
        idx = indices.detach().long().to(self.device)
        val = h_new.detach().float().to(self.device).clamp(0.0, 1.0)
        old = self.h[idx]
        cnt = self.cnt[idx]
        m = self.momentum

        # 如果是第一次遇到该样本，直接赋值；否则进行 EMA 更新
        fresh = (cnt < 0.5)
        out = torch.where(fresh, val, (1 - m) * old + m * val)

        self.h[idx] = out
        self.cnt[idx] = cnt + 1

    @torch.no_grad()
    def get(self, indices: torch.Tensor):
        idx = indices.detach().long().to(self.device)
        return self.h[idx]

class HardnessEstimator(nn.Module):
    """
    适配 LNLN 的难度估计器 (去除缺失先验版)。
    仅融合：
    1. Direct (重构难度)
    2. Indirect (模态不一致性)
    3. Task (预测误差)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # 权重系数 (不再需要 alpha_miss)
        self.alpha_direct = float(cfg.get("alpha_direct", 0.45))   # 建议稍微调高重构权重
        self.alpha_indirect = float(cfg.get("alpha_indirect", 0.30))
        self.alpha_task = float(cfg.get("alpha_task", 0.25))       # 建议稍微调高任务权重

        self.beta = float(cfg.get("beta", 2.5))
        self.topk_ratio = float(cfg.get("topk_ratio", 0.5))
        self.use_cos = bool(cfg.get("use_cos", True))

        # 为每个指标实例化归一化器
        self.norm_direct = RunningNorm()
        self.norm_indirect = RunningNorm()
        self.norm_task = RunningNorm()

    def _direct(self, out: dict, data: dict):
        """
        计算直接难度（重构误差）。
        逻辑：对比原始完整特征与重构特征，重点关注 Top-K 误差大的 Token。
        """
        if out.get("rec_feats") is None or out.get("complete_feats") is None:
            return torch.zeros(out["sentiment_preds"].shape[0], device=out["sentiment_preds"].device)

        rec = out["rec_feats"]
        comp = out["complete_feats"]
        # LNLN 的特征拼接顺序是 [Audio, Vision, Language]
        B, T, _ = rec.shape
        tok = T // 3 
        
        rec_a, rec_v, rec_l = rec[:, 0:tok], rec[:, tok:2*tok], rec[:, 2*tok:3*tok]
        cmp_a, cmp_v, cmp_l = comp[:, 0:tok], comp[:, tok:2*tok], comp[:, 2*tok:3*tok]

        def agg(x_rec, x_cmp):
            # 1. MSE
            mse = (x_rec - x_cmp).pow(2).mean(dim=-1) # (B, 8)
            # 2. Top-K Mean
            k = max(1, int(self.topk_ratio * tok))
            score = torch.topk(mse, k=k, dim=1).values.mean(dim=1)
            # 3. Cosine Distance
            if self.use_cos:
                cos = F.cosine_similarity(x_rec, x_cmp, dim=-1)
                cos_d = (1.0 - cos).clamp(0.0, 2.0)
                score = score + 0.5 * torch.topk(cos_d, k=k, dim=1).values.mean(dim=1)
            return score

        s_a, s_v, s_l = agg(rec_a, cmp_a), agg(rec_v, cmp_v), agg(rec_l, cmp_l)
        
        # 动态加权：缺失率越高的模态，其重构误差在总分中占比越大
        mr_a = data["labels"]["missing_rate_a"].to(s_a.device).view(-1)
        mr_v = data["labels"]["missing_rate_v"].to(s_a.device).view(-1)
        mr_l = data["labels"]["missing_rate_l"].to(s_a.device).view(-1)
        
        w_a, w_v, w_l = 1.0 + mr_a, 1.0 + mr_v, 1.0 + mr_l
        direct = (w_a * s_a + w_v * s_v + w_l * s_l) / (w_a + w_v + w_l + 1e-6)
        
        return direct

    def _indirect(self, out: dict):
        """
        计算间接难度（模态不一致性）。
        逻辑：计算单模态特征两两之间的余弦距离。
        """
        # 注意：需要确保 models/lnln.py 返回了 h_1_*
        if out.get("h_1_a") is None:
            return torch.zeros(out["sentiment_preds"].shape[0], device=out["sentiment_preds"].device)

        ha = out["h_1_a"].mean(dim=1)
        hv = out["h_1_v"].mean(dim=1)
        hl = out["h_1_l"].mean(dim=1)

        def cos_d(x, y): 
            return (1.0 - F.cosine_similarity(x, y, dim=-1)).clamp(0.0, 2.0)

        d_av = cos_d(ha, hv)
        d_al = cos_d(ha, hl)
        d_vl = cos_d(hv, hl)

        return (d_av + d_al + d_vl) / 3.0

    def _task(self, out: dict, label: dict):
        """计算任务难度（预测误差）"""
        pred = out["sentiment_preds"].view(-1)
        y = label["sentiment_labels"].view(-1)
        return (pred - y).pow(2)

    def forward(self, out: dict, label: dict, data: dict, is_train: bool = True):
        """
        前向传播计算综合难度。
        不再包含 _miss 计算。
        """
        # 1. 计算各项原始指标
        direct = self._direct(out, data)
        indirect = self._indirect(out)
        task = self._task(out, label)

        # 2. 更新归一化统计量 (仅训练时)
        if is_train:
            with torch.no_grad():
                self.norm_direct.update(direct.detach())
                self.norm_indirect.update(indirect.detach())
                self.norm_task.update(task.detach())

        # 3. 归一化并映射到 (0, 1)
        direct_n = torch.sigmoid(self.beta * self.norm_direct.normalize(direct))
        indirect_n = torch.sigmoid(self.beta * self.norm_indirect.normalize(indirect))
        task_n = torch.sigmoid(self.beta * self.norm_task.normalize(task))

        # 4. 加权求和 (已移除 miss 项)
        h = (
            self.alpha_direct * direct_n +
            self.alpha_indirect * indirect_n +
            self.alpha_task * task_n
        )
        
        # 返回 clamp 后的综合难度，以及原始分量用于 Scheduler 的 Split Source 逻辑
        return h.clamp(0.0, 1.0), {
            "direct": direct,
            "indirect": indirect,
            "task": task
        }


def _rank01(x: torch.Tensor):
    """计算 Rank [0, 1]"""
    B = x.numel()
    if B <= 1: return torch.zeros_like(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(B, device=x.device, dtype=torch.float)
    return ranks / float(B - 1)


class AdaptiveHardnessScheduler:
    """
    自适应调度器。
    根据综合难度或独立难度源，生成样本权重和门控值。
    """
    def __init__(self, cfg: dict, total_epochs: int):
        self.total_epochs = total_epochs
        
        self.warmup_epochs = int(cfg.get("warmup_epochs", 5))
        self.q_start = float(cfg.get("q_start", 0.3))
        self.q_end = float(cfg.get("q_end", 1.0))
        self.temp_start = float(cfg.get("temp_start", 0.15))
        self.temp_end = float(cfg.get("temp_end", 0.05))

        self.eta_max = float(cfg.get("eta_max", 0.6))
        self.p_pred = float(cfg.get("p_pred", 2.0))
        self.p_rec = float(cfg.get("p_rec", 1.5))
        
        self.w_clip_min = float(cfg.get("w_clip_min", 0.01))
        self.w_clip_max = float(cfg.get("w_clip_max", 5.0))
        
        self.use_split_sources = bool(cfg.get("use_split_sources", True))

    def _progress(self, epoch):
        if epoch <= self.warmup_epochs: return 0.0
        return float(epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)

    def _anneal(self, a, b, t): return a + (b - a) * t

    @torch.no_grad()
    def map(self, h_all, epoch, h_direct=None, h_task=None):
        device = h_all.device
        prog = self._progress(epoch)
        
        # Warmup 期全通过
        if prog <= 0.0:
            ones = torch.ones_like(h_all)
            return ones, ones

        q = self._anneal(self.q_start, self.q_end, prog)
        temp = self._anneal(self.temp_start, self.temp_end, prog)
        eta = self.eta_max * prog

        # 选择驱动源
        if self.use_split_sources and (h_direct is not None) and (h_task is not None):
            src_pred = h_task    # 预测任务权重主要看预测误差
            src_rec = h_direct   # 重构任务权重主要看重构误差
        else:
            src_pred = h_all
            src_rec = h_all

        # Gating (门控)
        tau_pred = torch.quantile(src_pred, q=q).item()
        g_pred = torch.sigmoid((tau_pred - src_pred) / max(temp, 1e-6))
        
        # Reweighting (重加权) - 越难权重越大
        r_pred = _rank01(src_pred)
        r_rec = _rank01(src_rec)

        w_pred = (1.0 - eta) + eta * (r_pred.clamp(1e-6, 1.0) ** self.p_pred)
        w_rec = (1.0 - eta) + eta * (r_rec.clamp(1e-6, 1.0) ** self.p_rec)

        # 组合并裁剪
        w_pred = (w_pred * g_pred).clamp(self.w_clip_min, self.w_clip_max)
        
        # 重构任务通常不需要严格门控，因为高缺失样本重构误差必然大，需要模型去学
        w_rec = w_rec.clamp(self.w_clip_min, self.w_clip_max)

        # Mean-Preserving
        w_pred = w_pred / (w_pred.mean() + 1e-6)
        w_rec = w_rec / (w_rec.mean() + 1e-6)

        return w_pred, w_rec