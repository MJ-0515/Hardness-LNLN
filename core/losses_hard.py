# core/losses.py
from torch import nn
from torch.nn import functional as F
class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ... (原初始化不变)
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.CE_Fn = nn.CrossEntropyLoss()
        
        # [修改] 使用 reduction='none' 以便后续手动加权
        self.MSE_Fn = nn.MSELoss(reduction='none') 

    def forward(self, out, label, sample_weight_pred=None, sample_weight_rec=None):
        # 1. Completeness Check Loss (辅助任务，不加权或保持原样)
        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']).mean() if out['w'] is not None else 0

        # 2. Adversarial Loss (辅助任务，保持原样)
        l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0

        # 3. Reconstruction Loss (支持 Curriculum 加权)
        if out['rec_feats'] is not None and out['complete_feats'] is not None:
            l_rec_vec = self.MSE_Fn(out['rec_feats'], out['complete_feats']).mean(dim=(1, 2)) # (B,)
            if sample_weight_rec is not None:
                l_rec = (l_rec_vec * sample_weight_rec).sum() / (sample_weight_rec.sum() + 1e-6)
            else:
                l_rec = l_rec_vec.mean()
        else:
            l_rec = 0

        # 4. Sentiment Prediction Loss (主任务，支持 Curriculum 加权)
        l_sp_vec = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels']).view(-1) # (B,)
        if sample_weight_pred is not None:
            l_sp = (l_sp_vec * sample_weight_pred).sum() / (sample_weight_pred.sum() + 1e-6)
        else:
            l_sp = l_sp_vec.mean()
        
        # 总 Loss
        loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_sp

        return {'loss': loss, 'l_sp': l_sp, 'l_cc': l_cc, 'l_adv': l_adv, 'l_rec': l_rec}