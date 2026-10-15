

import os
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import scipy.io as scio
import torch
from torch.optim import Adam
from torch.autograd import Variable
from utils import gradient
from testMat import showLossChart
from torch import nn
import copy

# from net import LRR_NET, Vgg16
# from net_nuclear import LRR_NET, Vgg16
from model import TwoBranchesFusionNet
from args import Args as args
import utils
import random
import torchvision.models as models
import pytorch_msssim
from GIFNetDataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--trainDataRoot', type=str, default='./train_data/GIFNet_Train', help='Root path of the train data')
parser.add_argument('--gate', action=argparse.BooleanOptionalAction, default=args.gate)
parser.add_argument('--teacher', action=argparse.BooleanOptionalAction, default=args.teacher)
parser.add_argument('--gate-loss-weight', type=float, default=args.gate_loss_weight)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=args.epochs)
parser.add_argument('--batch-size', type=int, default=args.batch_size)
parser.add_argument('--learning-rate', type=float, default=args.lr)
parser.add_argument('--train-num', type=int, default=args.train_num,
                    help='Exclusive upper index, preserving the original GIFNet loader convention')
parser.add_argument('--output-dir', default=args.save_fusion_model)
parser.add_argument('--confidence-beta', type=float, default=getattr(args, 'beta_conf', 2.0))
parser.add_argument('--pixel-threshold', type=float, default=getattr(args, 'qT_min', 0.97))
parser.add_argument('--quality-threshold', type=float, default=getattr(args, 'qT_min', 0.97))
parser.add_argument('--ema-decay', type=float, default=getattr(args, 'ema_alpha', 0.5))
parser.add_argument('--ep-mode', choices=['fusion', 'dual', 'confidence', 'full'], default='full',
                    help='EP component ablation; full preserves the original target and quality gate')
parser.add_argument('--permute-auxiliary', action='store_true',
                    help='Nonzero within-batch cyclic shift of auxiliary pairs only')
parser.add_argument('--profile-training', action='store_true',
                    help='Record complete CUDA update cost at fixed post-ramp batches, excluding gradient probes')

opt = parser.parse_args()
if opt.batch_size < 1 or opt.epochs < 1 or opt.train_num < 2:
    parser.error('batch-size and epochs must be positive; train-num must exceed1')
if not np.isfinite(opt.learning_rate) or opt.learning_rate <= 0:
    parser.error('learning-rate must be positive and finite')
if opt.confidence_beta <= 0 or not 0 <= opt.ema_decay < 1:
    parser.error('confidence-beta must be positive and ema-decay must be in [0,1)')
if not 0 <= opt.pixel_threshold <= 1 or not 0 <= opt.quality_threshold <= 1:
    parser.error('Both confidence thresholds must be in [0,1]')
args.gate, args.teacher = opt.gate, opt.teacher
args.gate_loss_weight = opt.gate_loss_weight
args.epochs, args.train_num = opt.epochs, opt.train_num
args.batch_size, args.lr = opt.batch_size, opt.learning_rate
args.save_fusion_model = opt.output_dir
args.save_loss_dir = os.path.join(opt.output_dir, 'loss')
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

EPSILON = 1e-5

# ---------------- default hyperparams for the improved scheme ----------------
BETA_CONF = opt.confidence_beta
ALPHA_FUSED = getattr(args, 'alpha_fused', 0.5)       # fused error weight between VI and IR
PIXEL_THRESHOLD = opt.pixel_threshold
QUALITY_THRESHOLD = opt.quality_threshold
LAMBDA_SELF_START = getattr(args, 'lambda_self_start', 0.06)
LAMBDA_SELF_END = getattr(args, 'lambda_self_end', 1.0)
LAMBDA_SELF_WARMUP = getattr(args, 'lambda_self_warmup', 1000)  # warmup iterations
EMA_DECAY = opt.ema_decay
REC_WEIGHT_CAP = getattr(args, 'rec_weight_cap', 0.6)  # cap on rec expert weight (optional)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Helper utilities for Teacher/EP -----------------
def compute_confidences_by_mse_multi(teacher_outputs, refs, beta=BETA_CONF, eps=EPSILON):
    outs = list(teacher_outputs)
    B, _, H, W = outs[0].shape
    errs = []
    for idx, out in enumerate(outs):
        # determine error reference logic:
        # assume index 0 = fused, index 1 = rec (matches our usage)
        if idx == 0:
            # fused: compare to both vi and ir
            vi = refs['vi']
            ir = refs['ir']
            err_vi = torch.mean((out - vi) ** 2, dim=1, keepdim=True)  # [B,1,H,W]
            err_ir = torch.mean((out - ir) ** 2, dim=1, keepdim=True)
            err = ALPHA_FUSED * err_vi + (1.0 - ALPHA_FUSED) * err_ir
        else:
            # rec: compare to visible only
            vi = refs['vi']
            err = torch.mean((out - vi) ** 2, dim=1, keepdim=True)
        errs.append(err)
    errs_stack = torch.cat(errs, dim=1)  # [B,K,H,W]
    conf = F.softmax(-beta * errs_stack, dim=1)  # [B,K,H,W]
    return conf

def ensemble_from_conf(teacher_outputs, conf_stack, eps=EPSILON):
    """
    teacher_outputs: list of [B,C,H,W] in same order as conf_stack channels
    conf_stack: [B,K,H,W]
    returns: ensembled [B,C,H,W], max_conf_map [B,H,W]
    """
    # convert Gate10O to [B,K,C,H,W]
    outs = torch.stack(teacher_outputs, dim=1)  # [B,K,C,H,W]
    w_expand = conf_stack.unsqueeze(2)  # [B,K,1,H,W]
    ens = torch.sum(w_expand * outs, dim=1)  # [B,C,H,W]
    max_conf, _ = torch.max(conf_stack, dim=1)  # [B,H,W]
    return ens, max_conf

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def auxiliary_permutation(batch_size, generator):
    indices = torch.arange(batch_size)
    if batch_size < 2:
        return indices
    shift = torch.randint(1, batch_size, (1,), generator=generator).item()
    return torch.roll(indices, shifts=shift)


def initialize_fusion_model(s, n, channel, stride, gate):
    """Align shared initial parameters for the local gate-on/off comparison."""
    gated = TwoBranchesFusionNet(s, n, channel, stride, True)
    if gate:
        return gated
    # Constructing the gate-disabled graph must not advance the canonical
    # initialization RNG beyond the gate-enabled reference state.
    with torch.random.fork_rng(devices=[]):
        backbone = TwoBranchesFusionNet(s, n, channel, stride, False)
    reference = gated.state_dict()
    backbone.load_state_dict({key: reference[key] for key in backbone.state_dict()}, strict=True)
    return backbone


def load_data(path, train_num):
    imgs_path, _ = utils.list_images(path)
    imgs_path = imgs_path[:train_num]
    random.shuffle(imgs_path)
    return imgs_path


def lossChartSave(temp_loss,lossName,lossList):
    # save item1_spe loss
    loss_filename_path = lossName + temp_loss
    save_loss_path = os.path.join(os.path.join(args.save_loss_dir), loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': lossList})
    showLossChart(save_loss_path,os.path.join(args.save_loss_dir)+"/"+lossName+'.png')


def main():
    densenet = models.densenet121(pretrained=True)
    densenet.eval()
    densenet = densenet.to(device)
    features = list(densenet.features.children())

    features_1 = nn.Sequential(*features[:4])
    features_2 = nn.Sequential(*features[4:6])
    features_3 = nn.Sequential(*features[6:8])
    features_4 = nn.Sequential(*features[8:10])
    features_5 = nn.Sequential(*features[10:11])

    twoBranchesFusionModel = initialize_fusion_model(args.s, args.n, args.channel, args.stride, args.gate)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError('DAHCFormer training requires a CUDA-enabled PyTorch installation and GPU.')
    if args.cuda:
        twoBranchesFusionModel.cuda()

    optimizer_TB = Adam(twoBranchesFusionModel.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim


    # ---------------- Teacher init ----------------
    if args.teacher:
        teacher_model = copy.deepcopy(twoBranchesFusionModel)
        for p in teacher_model.parameters():
            p.requires_grad = False
        if args.cuda:
            teacher_model.cuda()
    # ------------------------------------------------

    #------------ Begin Load Data
    root_dir = opt.trainDataRoot
    image_numbers = list(range(1, args.train_num))

    transform = transforms.Compose([
        transforms.ToTensor()  # 转为Tensor
    ])


    #CombinationAutoEncoder, TwoBranchesFusionNet

    custom_dataset = CustomDataset(root_dir, image_numbers, transform=transform)
    batch_size = args.batch_size
    sampler_generator = torch.Generator().manual_seed(opt.seed)
    auxiliary_generator = torch.Generator().manual_seed(opt.seed + 1000003)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True,
                             generator=sampler_generator)
    os.makedirs(args.save_fusion_model, exist_ok=True)
    config = {
        'seed': opt.seed, 'gate': args.gate, 'teacher': args.teacher,
        'epochs': args.epochs, 'batch_size': batch_size, 'learning_rate': args.lr,
        'sample_groups': len(custom_dataset),
        'outer_batches_per_epoch': len(data_loader),
        'planned_optimizer_updates': 2 * args.epochs * len(data_loader),
        'gate_loss_weight': args.gate_loss_weight, 'ema_decay': EMA_DECAY,
        'ep_mode': opt.ep_mode,
        'permute_auxiliary': opt.permute_auxiliary,
        'auxiliary_permutation_seed': opt.seed + 1000003,
        'auxiliary_pairing_rule': 'nonzero within-batch cyclic shift; singleton unchanged' if opt.permute_auxiliary else 'original paired samples',
        'profile_training': opt.profile_training,
        'initialization': 'canonical gate-enabled initialization; gate-off copies all shared state tensors',
        'beta': BETA_CONF, 'alpha': ALPHA_FUSED, 'gamma': REC_WEIGHT_CAP,
        'pixel_threshold': PIXEL_THRESHOLD, 'batch_quality_threshold': QUALITY_THRESHOLD,
        'sampler_seed': opt.seed, 'sampler': 'separate seeded torch.Generator',
        'warmup_outer_batches': LAMBDA_SELF_WARMUP,
        'ep_weight_start': LAMBDA_SELF_START, 'ep_weight_end': LAMBDA_SELF_END,
        'torch': torch.__version__, 'data_root': os.path.abspath(root_dir),
    }
    with open(os.path.join(args.save_fusion_model, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    #------------ End Load Data

    # tbar = trange(args.epochs)
    print('Start training.....')

    step = 10

    # creating save path
    temp_path_model = os.path.join(args.save_fusion_model)
    if os.path.exists(temp_path_model) is False:
        os.makedirs(temp_path_model, exist_ok=True)

    temp_path_loss = os.path.join(args.save_loss_dir)
    if os.path.exists(temp_path_loss) is False:
        os.makedirs(temp_path_loss, exist_ok=True)

    Loss_list_item1_spe = []
    Loss_list_item1_com = []
    Loss_list_item1_learning = []
    Loss_list_item2_spe = []
    Loss_list_item2_com = []
    Loss_list_item1_gate = []
    Loss_list_item2_gate = []

    Loss_list_all = []

    viz_index = 0
    loss_item1_spe = 0.
    loss_item1_com = 0.
    loss_item1_gate = 0.
    loss_item1_learning = 0.
    loss_item2_spe = 0.
    loss_item2_com = 0.
    loss_item2_gate = 0.
    twoBranchesFusionModel.train()
    global_step = 0
    ep_counters = {
        'eligible_outer_batches': 0, 'selected_outer_batches': 0,
        'nonzero_ep_loss_batches': 0, 'parameter_gradient_checks': 0,
        'nonzero_parameter_gradient_checks': 0,
    }
    run_started = time.perf_counter()
    auxiliary_counts = {'examples': 0, 'mismatched_examples': 0, 'singleton_batches': 0}
    training_profile = []
    for e in range(args.epochs):

        batch_num = len(data_loader)
        loss_item1_spe = 0.
        loss_item2_spe = 0.
        loss_item1_com = 0.
        loss_item2_com = 0.
        loss_item1_gate = 0.
        loss_item2_gate = 0.
        for idx, batch in enumerate(data_loader):
            if args.teacher:
                global_step += 1
            img_ir, img_vi, img_ir_NF, img_vi_FF = batch

            batch_ir = Variable(img_ir, requires_grad=False)
            batch_vi = Variable(img_vi, requires_grad=False)
            img_ir_NF = Variable(img_ir_NF, requires_grad=False)
            img_vi_FF = Variable(img_vi_FF, requires_grad=False)

            if args.cuda:
                batch_ir = batch_ir.cuda()
                batch_vi = batch_vi.cuda()
                img_ir_NF = img_ir_NF.cuda()
                img_vi_FF = img_vi_FF.cuda()

            auxiliary_counts['examples'] += batch_ir.shape[0]
            if opt.permute_auxiliary:
                permutation = auxiliary_permutation(batch_ir.shape[0], auxiliary_generator).to(batch_ir.device)
                aux_ir, aux_vi = batch_ir[permutation], batch_vi[permutation]
                aux_nf, aux_ff = img_ir_NF[permutation], img_vi_FF[permutation]
                auxiliary_counts['mismatched_examples'] += int((permutation != torch.arange(batch_ir.shape[0], device=batch_ir.device)).sum().item())
                auxiliary_counts['singleton_batches'] += int(batch_ir.shape[0] == 1)
            else:
                aux_ir, aux_vi, aux_nf, aux_ff = batch_ir, batch_vi, img_ir_NF, img_vi_FF

            IVIF_step = 1
            MFIF_step = 1
            outer_index = e * len(data_loader) + idx + 1
            profile_this_step = (opt.profile_training and args.cuda
                                 and 2101 <= outer_index <= 2200 and outer_index % 50 != 0)
            if profile_this_step:
                torch.cuda.synchronize()
                allocated_at_start = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                profile_start = torch.cuda.Event(enable_timing=True)
                profile_end = torch.cuda.Event(enable_timing=True)
                profile_start.record()

            #IVIF branch
            for _idx in range(IVIF_step):
                optimizer_TB.zero_grad()

                fea_com = twoBranchesFusionModel.forward_encoder(batch_ir,batch_vi)
                with torch.no_grad():
                    fea_com_mfif = twoBranchesFusionModel.forward_encoder(aux_nf,aux_ff)

                out_rec = twoBranchesFusionModel.forward_rec_decoder(fea_com)

                fea_fused = twoBranchesFusionModel.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com_mfif, trainingTag = 1)

                out_f = twoBranchesFusionModel.forward_mixed_decoder(fea_com, fea_fused)

                #计算源图像的信息量的度量。
                with torch.no_grad():
                    t_batch_ir = batch_ir.clone()
                    t_batch_vi = batch_vi.clone()
                    dup_ir = torch.cat([t_batch_ir,t_batch_ir,t_batch_ir],1)
                    dup_vi = torch.cat([t_batch_vi,t_batch_vi,t_batch_vi],1)

                    # -----------------获取CNN特征权重 开始------------------
                    layer1_feature_ir = features_1(dup_ir)
                    layer2_feature_ir = features_2(layer1_feature_ir)
                    layer3_feature_ir = features_3(layer2_feature_ir)
                    layer4_feature_ir = features_4(layer3_feature_ir)
                    layer5_feature_ir = features_5(layer4_feature_ir)

                    layer1_feature_vi = features_1(dup_vi)
                    layer2_feature_vi = features_2(layer1_feature_vi)
                    layer3_feature_vi = features_3(layer2_feature_vi)
                    layer4_feature_vi = features_4(layer3_feature_vi)
                    layer5_feature_vi = features_5(layer4_feature_vi)

                    layer1_feature_ir = gradient(layer1_feature_ir)**2
                    layer2_feature_ir = gradient(layer2_feature_ir)**2
                    layer3_feature_ir = gradient(layer3_feature_ir)**2
                    layer4_feature_ir = gradient(layer4_feature_ir)**2
                    layer5_feature_ir = gradient(layer5_feature_ir)**2
                    grad_ir_cnn = torch.mean(layer1_feature_ir)+torch.mean(layer2_feature_ir)+torch.mean(layer3_feature_ir)+torch.mean(layer4_feature_ir)+torch.mean(layer5_feature_ir)
                    grad_ir_cnn /= 5

                    layer1_feature_vi = gradient(layer1_feature_vi)**2
                    layer2_feature_vi = gradient(layer2_feature_vi)**2
                    layer3_feature_vi = gradient(layer3_feature_vi)**2
                    layer4_feature_vi = gradient(layer4_feature_vi)**2
                    layer5_feature_vi = gradient(layer5_feature_vi)**2
                    grad_vi_cnn = torch.mean(layer1_feature_vi) + torch.mean(layer2_feature_vi)+ torch.mean(layer3_feature_vi)+ torch.mean(layer4_feature_vi)+ torch.mean(layer5_feature_vi)
                    grad_vi_cnn /= 5

                    if args.cuda:
                        grad_ir_cnn = grad_ir_cnn.cuda(args.device)
                        grad_vi_cnn = grad_vi_cnn.cuda(args.device)

                    weightNonInterestedIR_cnn = torch.exp(grad_ir_cnn)/(torch.exp(grad_ir_cnn)+torch.exp(grad_vi_cnn))
                    weightNonInterestedVI_cnn = torch.exp(grad_vi_cnn)/(torch.exp(grad_ir_cnn)+torch.exp(grad_vi_cnn))

                    # -----------------获取CNN特征权重 结束--------------------


                item1_IM_loss_cnn = weightNonInterestedIR_cnn*mse_loss(out_f, batch_ir) + weightNonInterestedVI_cnn*mse_loss(out_f,batch_vi)
                item1_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi))
                item1_IM_loss =  item1_IM_loss_cnn + item1_commonLoss

                L_self = torch.tensor(0.0, device=device)
                qT = torch.tensor(0.0, device=device)
                # teacher forward (no grad) and ensemble pseudo target construction
                if args.teacher:
                    with torch.no_grad():
                        # run teacher on the same inputs
                        teacher_fea_com = teacher_model.forward_encoder(batch_ir, batch_vi)
                        # Teacher and student use the same auxiliary pairing.
                        teacher_fea_com_mfif = teacher_model.forward_encoder(aux_nf, aux_ff)
                        teacher_fea_fused = teacher_model.forward_MultiTask_branch(fea_com_ivif=teacher_fea_com,
                                                                                   fea_com_mfif=teacher_fea_com_mfif,
                                                                                   trainingTag=1)
                        teacher_out_fused = teacher_model.forward_mixed_decoder(teacher_fea_com, teacher_fea_fused)  # [B,C,H,W]
                        teacher_out_rec = teacher_model.forward_rec_decoder(teacher_fea_com)  # reconstruction head
                        # compute confidences by MSE to visible (batch_vi) -- shape [B,K,H,W]
                        teacher_list = [teacher_out_fused, teacher_out_rec]
                        refs = {'vi': batch_vi, 'ir': batch_ir}
                        conf_stack = compute_confidences_by_mse_multi(teacher_list, refs, beta=BETA_CONF)  # [B,K,H,W]
                        # optional cap to avoid REC dominating
                        try:
                            conf_stack[:, 1, :, :] = torch.clamp(conf_stack[:, 1, :, :], max=REC_WEIGHT_CAP)
                            conf_stack = conf_stack / (torch.sum(conf_stack, dim=1, keepdim=True) + 1e-8)
                        except Exception:
                            # if shape mismatch, skip cap
                            pass
                        # ensemble soft average
                        pseudo_target, max_conf_map = ensemble_from_conf(teacher_list, conf_stack)
                        if opt.ep_mode == 'fusion':
                            pseudo_target = teacher_out_fused
                            max_conf_map = torch.ones_like(max_conf_map)
                        elif opt.ep_mode == 'dual':
                            pseudo_target = (teacher_out_fused + teacher_out_rec) / 2.0
                            max_conf_map = torch.full_like(max_conf_map, 0.5)
                        max_conf_map_vals = max_conf_map
                        # compute qT (fraction of pixels with max_conf > TAU)
                        mask_high = (max_conf_map_vals > PIXEL_THRESHOLD).float()
                        qT = torch.mean(mask_high)  # scalar in [0,1]
                    # self-supervised loss (only apply when pseudo-target quality sufficient)
                    lambda_self = 0.0
                    if global_step > LAMBDA_SELF_WARMUP:
                        # linear ramp from start to end over warmup window
                        ramp = min(1.0, float(global_step - LAMBDA_SELF_WARMUP) / float(max(1, LAMBDA_SELF_WARMUP)))
                        lambda_self = LAMBDA_SELF_START + (LAMBDA_SELF_END - LAMBDA_SELF_START) * ramp
                    else:
                        lambda_self = 0.0
                    ep_selected = (opt.ep_mode != 'full' or qT.item() > QUALITY_THRESHOLD) and (lambda_self > 0.0)
                    if ep_selected:
                        # use max_conf_map as pixel-wise weight (expand to channels)
                        w_ens = max_conf_map.unsqueeze(1)  # [B,1,H,W]
                        # normalize per-sample to [0,1]
                        w_min = torch.amin(w_ens.view(w_ens.size(0), -1), dim=1).view(-1, 1, 1, 1)
                        w_max = torch.amax(w_ens.view(w_ens.size(0), -1), dim=1).view(-1, 1, 1, 1)
                        denom = (w_max - w_min) + 1e-8
                        w_ens = (w_ens - w_min) / denom
                        if opt.ep_mode in ('fusion', 'dual'):
                            # Uniform-target ablations use ordinary MSE: a
                            # constant confidence map would normalize to zero.
                            L_self = torch.mean((out_f - pseudo_target) ** 2)
                        else:
                            L_self = torch.mean(w_ens * (out_f - pseudo_target) ** 2)
                    else:
                        L_self = torch.tensor(0.0, device=device)
                    if global_step > LAMBDA_SELF_WARMUP:
                        ep_counters['eligible_outer_batches'] += 1
                    ep_counters['selected_outer_batches'] += int(ep_selected)
                    ep_counters['nonzero_ep_loss_batches'] += int(ep_selected and L_self.item() > 0)
                ep_grad_norm = None
                if args.teacher and ep_selected and global_step % 50 == 0:
                    final_weight = twoBranchesFusionModel.cnnDecoder.decoder[-2].weight
                    ep_grad = torch.autograd.grad(lambda_self * L_self, final_weight,
                                                  retain_graph=True)[0]
                    ep_grad_norm = ep_grad.norm().item()
                    ep_counters['parameter_gradient_checks'] += 1
                    ep_counters['nonzero_parameter_gradient_checks'] += int(ep_grad_norm > 0)
                if args.gate:
                    # 获取门控损失并加入总损失-mjl
                    item1_gate_loss = twoBranchesFusionModel.get_gate_loss()
                    if args.teacher:
                        total_loss_ivif = item1_IM_loss + args.gate_loss_weight * item1_gate_loss + lambda_self * L_self
                    else:
                        total_loss_ivif = item1_IM_loss + args.gate_loss_weight * item1_gate_loss
                    total_loss_ivif.backward()
                else:
                    if args.teacher:
                        total_loss_ivif = item1_IM_loss + lambda_self * L_self
                        total_loss_ivif.backward()
                    else:
                        total_loss_ivif = item1_IM_loss
                        total_loss_ivif.backward()
                optimizer_TB.step()
                # EMA update teacher after student step
                if args.teacher:
                    with torch.no_grad():
                        for t_param, s_param in zip(teacher_model.parameters(), twoBranchesFusionModel.parameters()):
                            t_param.data.mul_(EMA_DECAY).add_(s_param.data * (1.0 - EMA_DECAY))

            # Accumulate logging scalars only; retaining graph-bearing losses here
            # causes GPU memory to grow across the outer-batch loop.
            loss_item1_spe += item1_IM_loss_cnn.detach().item()
            loss_item1_com += item1_commonLoss.detach().item()
            if args.gate:
                loss_item1_gate += item1_gate_loss.detach().item()

            #MFIF branch
            for _idx in range(MFIF_step):
                optimizer_TB.zero_grad()


                fea_com = twoBranchesFusionModel.forward_encoder(img_ir_NF,img_vi_FF)
                with torch.no_grad():
                    fea_com_ivif = twoBranchesFusionModel.forward_encoder(aux_ir,aux_vi)

                out_rec = twoBranchesFusionModel.forward_rec_decoder(fea_com)

                fea_fused = twoBranchesFusionModel.forward_MultiTask_branch(fea_com_ivif = fea_com_ivif, fea_com_mfif = fea_com, trainingTag = 2)

                out_f = twoBranchesFusionModel.forward_mixed_decoder(fea_com, fea_fused)

                item2_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi))
                item2_supLoss = mse_loss(out_f,batch_vi)
                item2_clarity_loss = item2_supLoss + item2_commonLoss
                if args.gate:
                    # 获取门控损失并加入总损失-mjl
                    item2_gate_loss = twoBranchesFusionModel.get_gate_loss()
                    total_loss_mfif = item2_clarity_loss + args.gate_loss_weight * item2_gate_loss
                    total_loss_mfif.backward()
                else:
                    item2_clarity_loss.backward()
                optimizer_TB.step()
                # also update teacher on MFIF supervised step (EMA) - optional, keep teacher synced
                if args.teacher:
                    with torch.no_grad():

                        for t_param, s_param in zip(teacher_model.parameters(), twoBranchesFusionModel.parameters()):
                            t_param.data.mul_(EMA_DECAY).add_(s_param.data * (1.0 - EMA_DECAY))


            if profile_this_step:
                profile_end.record()
                profile_end.synchronize()
                peak_allocated = torch.cuda.max_memory_allocated()
                training_profile.append({
                    'outer_batch': outer_index, 'batch_size': int(batch_ir.shape[0]),
                    'cuda_elapsed_ms': profile_start.elapsed_time(profile_end),
                    'allocated_at_start_mib': allocated_at_start / 2**20,
                    'peak_allocated_mib': peak_allocated / 2**20,
                    'peak_increment_mib': (peak_allocated - allocated_at_start) / 2**20,
                    'ep_selected': bool(ep_selected) if args.teacher else False,
                })

            loss_item2_spe += item2_supLoss.detach().item()
            loss_item2_com += item2_commonLoss.detach().item()
            if args.gate:
                loss_item2_gate += item2_gate_loss.detach().item()

            if (idx + 1) % step == 0:
                measured_ivif = item1_IM_loss.detach().item()
                measured_mfif = item2_clarity_loss.detach().item()
                if args.gate:
                    measured_ivif += args.gate_loss_weight * item1_gate_loss.detach().item()
                    measured_mfif += args.gate_loss_weight * item2_gate_loss.detach().item()
                ep_loss = L_self.detach().item() if args.teacher else 0.0
                ep_weight = float(lambda_self) if args.teacher else 0.0
                measured_ivif += ep_weight * ep_loss
                record = {
                    'epoch': e + 1, 'outer_batch': e * len(data_loader) + idx + 1,
                    'optimizer_updates': 2 * (e * len(data_loader) + idx + 1),
                    'ivif_total_loss': measured_ivif, 'mfif_total_loss': measured_mfif,
                    'ep_loss': ep_loss, 'ep_weight': ep_weight,
                    'ep_quality_fraction': qT.detach().item() if args.teacher else 0.0,
                    'ep_last_decoder_weight_grad_norm': ep_grad_norm,
                    'ep_cumulative': dict(ep_counters),
                }
                with open(os.path.join(args.save_fusion_model, 'training_metrics.jsonl'),
                          'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')

            if idx % step == 0:

                loss_item1_spe /= step
                loss_item2_spe /= step
                loss_item1_com /= step
                loss_item2_com /= step
                if args.gate:
                    loss_item1_gate /= step
                    loss_item2_gate /= step


                mesg = "{}\t Count {} \t Epoch {}/{} \t Batch {}/{} \n " \
                        "IM loss: {:.6f} \n". \
                    format(time.ctime(), idx, e + 1, args.epochs, idx + 1, batch_num, item1_IM_loss_cnn.item())
                print(mesg)

                Loss_list_item1_spe.append(float(loss_item1_spe))
                Loss_list_item1_com.append(float(loss_item1_com))
                Loss_list_item2_spe.append(float(loss_item2_spe))
                Loss_list_item2_com.append(float(loss_item2_com))
                if args.gate:
                    Loss_list_item1_gate.append(float(loss_item1_gate))
                    Loss_list_item2_gate.append(float(loss_item2_gate))

                loss_item1_spe = 0.
                loss_item2_spe = 0.
                loss_item1_com = 0.
                loss_item2_com = 0.
                if args.gate:
                    loss_item1_gate = 0.
                    loss_item2_gate = 0.


            if (idx+1) % 300 == 0:
                temp_loss = "epoch_" + str(e + 1) + "_batch_" + str(idx + 1) + \
                            "_block_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                lossChartSave(temp_loss,"item1_spe_loss",Loss_list_item1_spe)
                lossChartSave(temp_loss,"item1_com_loss",Loss_list_item1_com)
                lossChartSave(temp_loss,"item2_spe_loss",Loss_list_item2_spe)
                lossChartSave(temp_loss,"item2_com_loss",Loss_list_item2_com)
                if args.gate:
                    lossChartSave(temp_loss, "item1_gate_loss", Loss_list_item1_gate)
                    lossChartSave(temp_loss,"item2_gate_loss",Loss_list_item2_gate)


            if (idx+1) % 700 == 0:
                # save model ever 700 iter.
                #twoBranchesFusionModel.eval()
                twoBranchesFusionModel.cpu()

                save_model_filename = "MTFusion_net_epoch_" + str(e + 1) + "_count_" + str(idx+1) + "_twoBranches"  + ".model"
                save_model_path = os.path.join(temp_path_model, save_model_filename)
                torch.save(twoBranchesFusionModel.state_dict(), save_model_path)


                print('Saving model at ' + save_model_path + '......')
                ##############
                #twoBranchesFusionModel.train()
                if (args.cuda):
                    twoBranchesFusionModel.cuda()

        # save model
        twoBranchesFusionModel.eval()
        twoBranchesFusionModel.cpu()
        save_model_filename = "MTFusion_net" + "_epoch_" + str(e + 1) + "_twoBranches"  + ".model"
        save_model_path = os.path.join(temp_path_model, save_model_filename)
        if args.teacher:
            torch.save({
                'student': twoBranchesFusionModel.state_dict(),
                'teacher': teacher_model.state_dict()
            }, save_model_path)
        else:
            torch.save(twoBranchesFusionModel.state_dict(), save_model_path)
        ##############
        twoBranchesFusionModel.train()
        if (args.cuda):
            twoBranchesFusionModel.cuda()
        print("\nCheckpoint, trained model saved at: " + save_model_path)


    summary = {
        'status': 'completed', 'seed': opt.seed,
        'outer_batches': args.epochs * len(data_loader),
        'optimizer_updates': 2 * args.epochs * len(data_loader),
        'training_loop_wall_seconds': time.perf_counter() - run_started,
        'ep_cumulative': ep_counters,
        'auxiliary_pairing_counts': auxiliary_counts,
        'profiled_outer_batches': len(training_profile),
        'final_checkpoint': save_model_filename,
    }
    with open(os.path.join(args.save_fusion_model, 'run_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    if training_profile:
        durations = np.array([row['cuda_elapsed_ms'] for row in training_profile])
        profile = {
            'seed': opt.seed, 'gate': args.gate, 'teacher': args.teacher, 'ep_mode': opt.ep_mode,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'scope': 'CUDA-event elapsed interval for full IVIF+MFIF updates including teacher/EP/EMA when enabled; excludes data loading, transfers, file writes and gradient probes; not end-to-end wall time',
            'selection': 'outer batches2101..2200 excluding multiples of50;98 planned measurements',
            'sampled_batches': len(training_profile),
            'mean_step_ms': float(durations.mean()),
            'step_sample_sd_ms': float(durations.std(ddof=1)) if len(durations)>1 else None,
            'peak_allocated_mib': max(row['peak_allocated_mib'] for row in training_profile),
            'ep_selected_batches': sum(row['ep_selected'] for row in training_profile),
            'steps': training_profile,
        }
        with open(os.path.join(args.save_fusion_model, 'training_profile.json'), 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
    print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
    main()
