import torch


class Evaluator:

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {'recall': self.recall_at_k, 'ndcg': self.ndcg_at_k}

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

        self.maxk_eval = max(config['val_topk'])

    def calculate_pos_index(self, preds, labels, split):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        if split == 'val':
            assert preds.shape[
                1] == self.maxk_eval, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        elif split == 'test':
            assert preds.shape[
                1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        else:
            raise ValueError(f"Invalid split: {split}")

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            # for j in range(self.maxk):

            maxk = self.maxk if (split == 'test') else self.maxk_eval
            for j in range(maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels, split):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels, split)
        for metric in self.config['metrics']:
            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            # for k in self.config['topk']:
            for k in topk_list:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index,
                                                                    k)
        return results
