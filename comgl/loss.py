import torch

def auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()

def adaptive_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (weight*torch.square(1 - (pos_out - neg_out))).sum()

def log_rank_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out):
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
    return pos_loss + neg_loss

def info_nce_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

def complementary_loss(x_dict, edge_index_dict, edge_type, num_nodes, num_neg, device, model):
    u_type, v_type = edge_type[0], edge_type[2]
    edge_label_index, edge_label = edge_index_dict[edge_type], edge_index_dict[edge_type].edge_label
    logits = model.predictor[0](x_dict[u_type], x_dict[v_type], edge_label_index.to(device)).sigmoid()
    y_pred = F.gumbel_softmax(logits, tau=0.01, hard=True)[:, 1]
    acc = (y_pred.cpu() == edge_label).type(torch.float)  # 边的预测准确率
    weight = torch.ones(num_nodes[1], 1, dtype=torch.float) 
    weight = calculate_weight(weight, acc, edge_label_index.cpu(), loc=1)
    weight = weight.to(device)
    return weight