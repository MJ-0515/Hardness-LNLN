import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_SAFE"] = "True"
import torch
import yaml
import argparse
import time
from core.dataset import MMDataLoader
from core.losses_hard import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results, interval_time, get_parameter_number
from models.lnln import build_model
from core.metric import MetricsTop 
from tqdm import tqdm
# [新增导入]
from models.hardness import HardnessEstimator, HardnessBank, AdaptiveHardnessScheduler

start = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

# 运行前设置的seed和yaml路径传给opt
parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
opt = parser.parse_args()
print("-------------------------------------------------------------------------------")
print(opt)    #Namespace(config_file='configs/train_mosi.yaml', seed=1111)
print("-------------------------------------------------------------------------------")


def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print("************************train.yaml中的内容**************************")
    print(args) 

    #最终seed还是看train/eval.yaml中的'base'里seed的值
    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("-------------------------------------------------------------------------")
    print("seed is fixed to {}".format(seed))
    print("-------------------------------------------------------------------------")

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("-------------------------------------------------------------------------")
    print("ckpt root :", ckpt_root)
    print("-------------------------------------------------------------------------")

    model = build_model(args).to(device)


    print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'],
                                 weight_decay=args['base']['weight_decay'])
    
    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = MultimodalLoss(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    # 在构建 model 之后，optimizer 之前或之后初始化 Hardness 模块
    use_hard = args.get("hardness", {}).get("enable", False) # 开关
    hard_est, hard_bank, hard_sched = None, None, None

    if use_hard:
        print("Initializing Hardness-Aware Curriculum Learning...")
        hard_est = HardnessEstimator(args["hardness"]["estimator"]).to(device)
        # Bank 放在 CPU 即可，Data Size 需要通过 dataLoader 获取
        hard_bank = HardnessBank(
                                num_samples=len(dataLoader["train"].dataset),
                                momentum=float(args["hardness"]["bank"].get("momentum", 0.05)),
                                 device="cpu")
        hard_sched = AdaptiveHardnessScheduler(
            args["hardness"]["scheduler"], 
            total_epochs=args['base']['n_epochs']
            )

    print("----------------------------------------------开始训练！！！---------------------------------------------------------")
    for epoch in range(1, args['base']['n_epochs']+1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))

        #------------------------------！！！进入 train   ！！！--------------------------------
        # [修改] 传入 hardness 模块
        train(model, train_loader, optimizer, loss_fn, epoch, metrics,
              use_hard, hard_est, hard_bank, hard_sched)

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, 
                                                  seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, 
                                             seed, save_best_model=True)
        
        end_time = time.time()
        epoch_mins, epoch_secs = interval_time(start_time, end_time)
        #print(f'Current Best Test Results: {best_test_results}\n')
        print("Epoch: {}/{} | Current Best Test Results: {} | \n Time: {}m {}s".format(epoch, args['base']['n_epochs'],
                                                                                       best_test_results, epoch_mins,
                                                                                       epoch_secs))

        scheduler_warmup.step()

# [修改] train 函数签名
def train(model, train_loader, optimizer, loss_fn, epoch, metrics,
          use_hard=False, hard_est=None, hard_bank=None, hard_sched=None):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        # 数据搬运
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}
        
        # 1. 前向传播
        out = model(complete_input, incomplete_input)

        # 2. [新增] 计算难度权重
        w_pred, w_rec = None, None
        if use_hard:
            indices = data['index']
            with torch.no_grad(): # 难度计算不产生梯度
                # 计算当前 batch 难度
                h_now, h_parts = hard_est(out, label, data, is_train=True)
                
                # 更新记忆库 (Bank)
                hard_bank.update(indices, h_now)
                h_all = hard_bank.get(indices).to(device)
                
                # 分离 Task 和 Direct 难度用于不同的任务加权
                # 需要重新 normalize task/direct 分量
                beta = hard_est.beta
                h_direct = torch.sigmoid(beta * hard_est.norm_direct.normalize(h_parts["direct"]))
                h_task = torch.sigmoid(beta * hard_est.norm_task.normalize(h_parts["task"]))
                
                # 调度器生成权重
                w_pred, w_rec = hard_sched.map(h_all, epoch, h_direct, h_task)
        
        # 3. 计算 Loss (传入权重)
        loss = loss_fn(out, label, sample_weight_pred=w_pred, sample_weight_rec=w_rec)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')




def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    
    # print(f'Test Loss Epoch {epoch}: {loss_dict}')
    # print(f'Test Results Epoch {epoch}: {results}')

    return results


if __name__ == '__main__':
    main()
print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))


