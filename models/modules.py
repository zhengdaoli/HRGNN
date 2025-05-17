import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import edge_index_to_dense
from utils.utils import is_nan_inf




class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, pred, targets, res=None):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        # """
        loss = self.loss(pred, targets)
        accuracy = self._calculate_accuracy(pred, targets)
        return loss, accuracy

    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs, targets):
        correct = self._get_correct(outputs)
        return 100. * (correct == targets).sum().float() / targets.size(0) # True/True


class BinaryClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __name__(self):
        return "BinaryClassificationLoss"
    
    def forward(self, pred, targets, res=None):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        pred = pred.float()
        if pred.size().numel() == targets.size().numel():
            targets = targets.float().view(pred.shape)
        else:
            targets = F.one_hot(targets.long(), num_classes=2).float()
        return super().forward(pred, targets, res=res)

    def _get_correct(self, outputs):
        return outputs > 0.5


class MixDecoupleClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        if reduction is None:
            self.loss = nn.NLLLoss(weight=weight)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, weight=weight)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)

    def __name__(self):
        return "MixDecoupleClassificationLoss"

class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None, weight=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.CrossEntropyLoss(reduction=reduction,weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(weight=weight)
            
    def __name__(self):
        return "MulticlassClassificationLoss"

    def forward(self, pred, targets, res=None):
        """
        :param targets:
        :param outputs:
        :return: loss and accuracy values
        """
        if targets.size(0) != 1:
            targets = targets.squeeze()
            
        targets = targets.long()
            
        return super().forward(pred, targets, res=res)
    
    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        """
        :param targets:
        :param outputs:
        :return: a loss value
        """
        raise NotImplementedError()

class LinkPredictionLoss(nn.Module):
    def __init__(self):
        super(LinkPredictionLoss, self).__init__()
        
    def forward(self, pred, targets, res=None):
        loss = F.binary_cross_entropy_with_logits(pred, targets)
        
        return loss

class PriorLoss(nn.Module):
    def __init__(self):
        super(PriorLoss, self).__init__()
    
    def wasserstein_distance(self, mu, sigma):
        mu_0 = torch.zeros_like(mu)
        # sigma_0 = torch.ones_like(sigma)
        
        term1 = torch.abs(mu - mu_0)
        # L2 norm of sigma and sigma_0:
        term2 = torch.sqrt(torch.abs(sigma**2 - 1))
        
        return term1 + term2
    
    def kl_distance(self, mu, sigma):
        dis = - 0.5 * (1 + 2*sigma - mu**2 - torch.exp(sigma)**2).sum(1).mean()
        return dis

    def forward(self, mu, sigma):
        # minimize the KL distance between the prior and standard normal distribution:
        dis = self.kl_distance(mu, sigma)
        # dis = self.wasserstein_distance(mu, sigma).mean()
        
        return dis

class MatchLoss(nn.Module):
    def __init__(self):
        super(MatchLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, Am_a_list, pos_a_list, neg_a_list):
        
        pos_a_tensor = torch.stack(pos_a_list)
        if neg_a_list is not None:
            neg_a_tensor = torch.stack(neg_a_list)

        pos_loss = 0
        neg_loss = 0
        if not isinstance(Am_a_list, list):
            return torch.FloatTensor(Am_a_list).mean()
        
        for Am_z in Am_a_list:
            pos_loss += self.bce_loss(Am_z.expand_as(pos_a_tensor), pos_a_tensor).mean()
            # neg_loss = - self.bce_loss(Am_z.expand_as(neg_a_tensor), neg_a_tensor).mean()
        total_loss = pos_loss + neg_loss
        
        return total_loss


class HVOLoss(nn.Module):
    def __init__(self, weight=None):
        super(HVOLoss, self).__init__()
        self.class_loss = MulticlassClassificationLoss(weight)
        self.match_loss = MatchLoss()
        self.prior_loss = PriorLoss()
        self.link_loss = LinkPredictionLoss()
        
    def forward(self, pred_y, target, res:dict):
        # extract each part from res dict:
        Am_agre_list = res['Am_agre']
        pos_agre_list = res['pos_agre']
        neg_agre_list = res['neg_agre']
        mu = res['mean_v']
        sigma = res['std_v']
        target_edge_index = res['target_edge_index']
        Am_dense_adjs = res['Am_dense_adjs'][0]
        batch = res['batch']

        # 1. Classification LOSS:
        class_loss, acc = self.class_loss(pred_y, target)
        class_loss = class_loss.reshape(1)
        
        # 2. Match LOSS:
        if Am_agre_list is not None:
            match_loss = self.match_loss(Am_agre_list, pos_agre_list, neg_agre_list).reshape(1)
        else:
            match_loss = torch.FloatTensor([0.0]).to(pred_y.device)
            match_loss.requires_grad_(False)
            
        # 3. Prior LOSS:
        prior_loss = self.prior_loss(mu, sigma).reshape(1)
        
        # 4. Link LOSS:
        link_loss = 0
        target_adjs = []
        edge_cum_idx = 0
        for i, adj in enumerate(Am_dense_adjs):
            node_num = adj.shape[0]
            node_indices = torch.where(batch == i)[0]
            edge_indices = (target_edge_index[0] >= node_indices.min()) & (target_edge_index[0] <= node_indices.max())
            graph_edge_index = target_edge_index[:, edge_indices]
            graph_edge_index -= edge_cum_idx
            dense_target_edge_index = edge_index_to_dense(graph_edge_index, num_nodes=node_num)
            target_adjs.append(dense_target_edge_index)
            edge_cum_idx += node_num
            
        for i in range(len(target_adjs)):
            # check Am_dense_adjs and target_adjs are Nan:
            link_l = self.link_loss(Am_dense_adjs[i], target_adjs[i])
            link_loss += link_l
            
        link_loss = link_loss/len(target_adjs)
            
        return (class_loss, match_loss, prior_loss, link_loss), acc
        
class BinaryHVOLoss(HVOLoss):
    def __init__(self, weight=None):
        super(BinaryHVOLoss, self).__init__(weight)
        self.class_loss = BinaryClassificationLoss(weight)
        
        
class CovarianceResidualError(RegressionLoss):  # For Cascade "Correlation"
    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        _, _, graph_emb, errors = outputs

        errors_minus_mean = errors - torch.mean(errors, dim=0)
        activations_minus_mean = graph_emb - torch.mean(graph_emb, dim=0)

        # todo check against commented code
        cov_per_pattern = torch.zeros(errors.shape)

        cov_error = 0.
        for o in range(errors.shape[1]):  # for each output unit
            for i in range(errors.shape[0]):  # for each pattern
                cov_per_pattern[i, o] = errors_minus_mean[i, o]*activations_minus_mean[i, 0]

            cov_error = cov_error + torch.abs(torch.sum(cov_per_pattern[:, o]))

        #print(torch.mean(cov_per_pattern, dim=0), torch.mean(errors_minus_mean), torch.mean(graph_emb))

        '''
        activations_minus_mean = torch.sum(activations_minus_mean, dim=1)
        activations_minus_mean = torch.unsqueeze(activations_minus_mean, dim=1)

        activations_minus_mean = torch.t(activations_minus_mean)

        cov_per_pattern = torch.mm(activations_minus_mean, errors_minus_mean)

        cov_abs = torch.abs(cov_per_pattern)

        # sum over output "units"
        cov_error = torch.sum(cov_abs)
        '''

        # Minus --> maximization problem!
        return - cov_error


class NN4GMulticlassClassificationLoss(MulticlassClassificationLoss):

    def mse(self, ts, ys, return_sum):

        targets_oh = torch.zeros(ys.shape)
        ts = ts.unsqueeze(1)
        targets_oh.scatter_(1, ts, value=1.)  # src must not be specified
        ts = targets_oh

        if return_sum == True:
            return torch.sum(0.5 * (ts - ys) ** 2) / len(ts)
        else:
            return 0.5 * (ts - ys) ** 2 / len(ts)

    def forward(self, targets, *outputs):

        preds, _, _, _ = outputs

        # Try MSE
        loss = self.mse(targets, preds, return_sum=True)

        #loss = self.loss(preds, targets)

        accuracy = self._calculate_accuracy(preds, targets)
        return loss, accuracy


class DiffPoolMulticlassClassificationLoss(MulticlassClassificationLoss):
    """
    DiffPool - No Link Prediction Loss
    """

    def forward(self, targets, *outputs):
        preds, lp_loss, ent_loss = outputs

        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        loss = self.loss(preds, targets)
        accuracy = self._calculate_accuracy(preds, targets)
        return loss + lp_loss + ent_loss, accuracy
