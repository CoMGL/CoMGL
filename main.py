import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from base_options import BaseOptions
from comgl.model import *
from comgl.utils import *
from comgl.loss import *
from utils import *
from evaluate import Evaluator
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score
from utils import *



args = BaseOptions().get_arguments()
print(args)
if args.wandb:
    wandb.init(project='CoMGL')
    wandb.run.name = '' + time.strftime("-%b-%d-%H:%M", time.localtime())
    wandb.config.update(args)

if args.cuda and torch.cuda.is_available(): 
    gpus_to_use = assign_free_gpus()
    args.cuda_idx = int(gpus_to_use[0])
    args.device = torch.device(f'cuda:{args.cuda_idx}' if torch.cuda.is_available() else torch.device('cpu'))
else:
    args.device = torch.device('cpu')
device = args.device
set_seed(args)

if args.train_on_subgraph:
    train_loader, val_loader, test_loader = load_dataset(args)

u_type, v_type = args.u_type, args.v_type   # node types of main view
edge_type, rev_edge_type = args.edge_type, args.rev_edge_type
view_dict = args.view_dict
edge_type_aux_1, rev_edge_type_aux_1 = view_dict[1][0], view_dict[1][1]
edge_type_aux_2, rev_edge_type_aux_2 = view_dict[2][0], view_dict[2][1]

model = CoMGL(
    args,
    view_dict=view_dict,
    predictor_name=args.predictor_name,
    lr=args.lr,
    dropout=args.dropout,
    grad_clip_norm=args.grad_clip_norm,
    gnn_num_layers=args.gnn_num_layers,
    aggregator_name=args.aggregator_name,
    mlp_num_layers=args.mlp_num_layers,
    gnn_hidden_channels=args.gnn_hidden_channels,
    agg_hidden_channels=args.agg_hidden_channels,
    mlp_hidden_channels=args.mlp_hidden_channels,  
    optimizer_name=args.optimizer,
    device=args.device
)

def link_prediction_train(train_loader):
    loss_log = []
    auc_score, ap_score = [], []
    model.train()
    for train_data in tqdm(train_loader, leave=True):
        model.optimizer.zero_grad()
        train_data = train_data.to(device)
        # 构建mask 数据集
        train_edge, val_edge, test_edge = RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.0,
                    neg_sampling_ratio=1,
                    edge_types=[edge_type_aux_1, edge_type_aux_2],
                    rev_edge_types=[rev_edge_type_aux_1, rev_edge_type_aux_2],
                )(train_data)
        x_dict, edge_index_dict = train_edge.x_dict, train_edge.edge_index_dict
        z = model.encode_aux(x_dict, edge_index_dict)
        u_emb = z[0]['openid'] + z[1]['openid']
        u_emb = torch.concat([u_emb, x_dict['openid']], dim=1)
        v_emb = train_data['project'].x

        aux_loss = 0
        for idx, aux_view in enumerate(view_dict[1:]):
            aux_edge_type = aux_view[0]
            aux_v_type = aux_edge_type[2]
            aux_edge_index = val_edge[aux_edge_type].edge_label_index.to(device)
            aux_y_true = val_edge[aux_edge_type].edge_label.to(device)
            aux_v_emb = z[idx][aux_v_type]
            aux_y_pred = model.aux_predictor[idx](u_emb, aux_v_emb, aux_edge_index).squeeze()
            aux_loss += F.binary_cross_entropy_with_logits(aux_y_pred, aux_y_true)

        # complementary loss
        complementarity_loss = complementarity_loss(z[0]['openid'], z[1]['openid'])
        
        edge_index = train_data[edge_type].edge_index
        neg_edge_index = negative_sampling(edge_index, num_nodes=(train_data['openid'].x.shape[0], train_data['project'].x.shape[0]))
        y_true = torch.concat([torch.ones_like(edge_index[0, :]), torch.zeros_like(neg_edge_index[0, :])]).float()
        all_edge_index = torch.concat([edge_index, neg_edge_index], dim=1)
        y_pred = model.predictor(u_emb, v_emb, all_edge_index).squeeze()
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true) + aux_loss + complementarity_loss
        loss.backward()
        model.optimizer.step()
        loss_log.append(loss.item())



@torch.no_grad()
def link_prediction_test(test_loader):
    auc_score, ap_score = [], []
    model.eval()
    for test_data in tqdm(test_loader, leave=True):
        test_data = test_data.to(device)
        x_dict, edge_index_dict = test_data.x_dict, test_data.edge_index_dict
        z = model.encode_aux(x_dict, edge_index_dict)
        u_emb = z[0][u_type] + z[1][u_type]
        u_emb = torch.concat([u_emb, x_dict[u_type]], dim=1)
        v_emb = test_data[v_type].x
        edge_index = test_data[edge_type].edge_index
        neg_edge_index = negative_sampling(edge_index, num_nodes=(test_data[u_type].x.shape[0], test_data[v_type].x.shape[0]))
        y_true = torch.concat([torch.ones_like(edge_index[0, :]), torch.zeros_like(neg_edge_index[0, :])]).float()
        all_edge_index = torch.concat([edge_index, neg_edge_index], dim=1)
        y_pred = model.predictor(u_emb, v_emb, all_edge_index).squeeze()
        auc_score.append(roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))
        ap_score.append(average_precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))


# def node_prediction_train(train_data):
#     model.train()
#     pos_out, neg_out = model(train_data)
#     num_neg = neg_out.size(0)
#     loss = calculate_loss(pos_out, neg_out, num_neg, margin=None)
#     return loss

# @torch.no_grad()
# def node_prediction_test(test_data):
#     model.eval()
#     pos_pred, neg_pred = model(test_data)
#     results = evaluate_rocauc(
#         evaluator,
#         pos_pred,
#         neg_pred)
#     return results

for epoch in range(20):
    results = link_prediction_train(train_loader)
    print(results)