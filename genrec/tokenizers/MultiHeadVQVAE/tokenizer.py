import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.tokenizer import SemIDTokenizer
from genrec.tokenizers.MultiHeadVQVAE.layers import MultiHeadVQVAE


class EmbDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class MultiHeadVQVAETokenizer(SemIDTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(MultiHeadVQVAETokenizer, self).__init__(config, dataset)

        self.id2item = dataset.id_mapping['id2item']

        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = int(np.sum(self.codebook_sizes) + 1)
        self.ignored_label = -100

    @property
    def n_digit(self):
        return self.config['vq_n_codebooks']

    @property
    def codebook_sizes(self):
        if isinstance(self.config['vq_codebook_size'], list):
            return self.config['vq_codebook_size']
        else:
            return [self.config['vq_codebook_size']] * self.n_digit

    @torch.no_grad()
    def _valid_collision(self, vqvae_model, dataloader):
        vqvae_model.eval()

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(dataloader):
            num_sample += len(data)
            data = data.to(self.config['device'])
            indices = vqvae_model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set))) / num_sample

        return collision_rate

    def _train_vqvae_epoch(self, model: MultiHeadVQVAE, train_data,
                           optimizer: torch.optim.Optimizer):
        model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        total_count = 0

        for batch_idx, data in enumerate(train_data):
            data = data.to(self.config['device'])
            optimizer.zero_grad()
            out, vq_loss, indices, unused_cnt = model(data)
            loss, loss_recon, quant_loss = model.compute_loss(out,
                                                              vq_loss,
                                                              xs=data)

            if torch.isnan(loss):
                raise ValueError("Training loss is nan")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += quant_loss.item()
            total_count += unused_cnt

        return total_loss, total_recon_loss, total_quant_loss, total_count

    def _train_vqvae(self, sent_embs: torch.Tensor, model_path: str):
        device = self.config['device']

        data = EmbDataset(sent_embs.cpu())

        vqvae_model = MultiHeadVQVAE(
            in_dim=data.dim,
            num_emb_list=self.codebook_sizes,
            e_dim=self.config['vqvae_e_dim'],
            layers=self.config['vqvae_hidden_sizes'],
            dropout_prob=self.config['vqvae_dropout'],
            bn=self.config['vqvae_bn'],
            loss_type='mse',
            quant_loss_weight=self.config['vqvae_quant_weight'],
            beta=self.config['vqvae_beta'],
            kmeans_init=self.config['kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            sk_epsilons=[0, 0, 0, self.config['vqvae_sk_epsilon']],
            sk_iters=self.config['vqvae_sk_iters']).to(device)
        self.log(vqvae_model)

        # Model training
        batch_size = self.config['vqvae_batch_size']
        num_epochs = self.config['vqvae_epoch']
        verbose = self.config['vqvae_verbose']

        
        vqvae_l2 = float(self.config['vqvae_l2'])
        if self.config['vqvae_optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(vqvae_model.parameters(),
                                         lr=self.config['vqvae_lr'],
                                         weight_decay=vqvae_l2)
        elif self.config['vqvae_optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(vqvae_model.parameters(),
                                          lr=self.config['vqvae_lr'],
                                          weight_decay=vqvae_l2)
        else:
            raise ValueError("Optimizer not supported")
        # optimizer = torch.optim.Adagrad(vqvae_model.parameters(),
        #                                 lr=self.config['vqvae_lr'])
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)

        best_loss = float('inf')
        best_collision_rate = float('inf')
        best_collision_epoch = 0

        self.log("[TOKENIZER] Training MultiVQ-VAE model...")

        vqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_recon_loss, train_quant_loss, total_count = self._train_vqvae_epoch(
                vqvae_model, dataloader, optimizer)

            if (epoch + 1) % verbose == 0:
                collision_rate = self._valid_collision(vqvae_model, dataloader)

                if self.config['vqvae_save_method'] == 'loss':
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_collision_epoch = epoch + 1
                        torch.save(vqvae_model.state_dict(), model_path)
                elif self.config['vqvae_save_method'] == 'collision':
                    if collision_rate < best_collision_rate:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_collision_epoch = epoch + 1
                        torch.save(vqvae_model.state_dict(), model_path)
                else:
                    raise ValueError("Save method not supported")

                self.log(
                    f"[TOKENIZER] VQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {train_loss}\n"
                    f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {train_recon_loss}\n"
                    f"\t  Quantization loss: {train_quant_loss}\n"
                    f"\t  Collision rate: {collision_rate}\n")

        self.log("[TOKENIZER] VQ-VAE training complete.")

        self.log(
            f"Best Epoch: {best_collision_epoch} Best Collision Rate: {best_collision_rate} Best Loss: {best_loss}"
        )

        vqvae_model.load_state_dict(torch.load(model_path))
        return vqvae_model

    def _check_collision(self, str_sem_ids):
        tot_item = len(str_sem_ids)
        tot_ids = len(set(str_sem_ids.tolist()))
        self.log(
            f'[TOKENIZER] Collision rate: {(tot_item - tot_ids) / tot_item}')
        return tot_item == tot_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        if self.config['sem_ids_path'] is None:
            sem_ids_path = os.path.join(self.config['result_dir'],
                                        self.config['sem_ids_dir'],
                                        f"{self.config['run_time']}.sem_ids")
            os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        else:
            sem_ids_path = os.path.join(
                self.config['base_result_dir'],
                f"{self.config['dataset']}_{self.config['category']}",
                f"TokenModel_Seq-{self.config['max_item_seq_len']}_MultiHeadVQVAE",
                self.config['sem_ids_dir'], self.config['sem_ids_path'])

        if not os.path.exists(sem_ids_path):
            self.log(f"{sem_ids_path} not found. Generating semantic IDs...")

            sent_embs = self.load_sent_emb(dataset)
            self.log(
                f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)

            self.log(
                f'[TOKENIZER] Semantic IDs not found. Training VQ-VAE model...'
            )
            embs_for_training = torch.FloatTensor(
                sent_embs[training_item_mask]).to(self.config['device'])
            sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])

            model_path = os.path.join(self.config['result_dir'],
                                      self.config['ckpt_dir'],
                                      f"{self.config['run_time']}.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            vqvae_model = self._train_vqvae(embs_for_training, model_path)
            self._generate_semantic_id(vqvae_model, sent_embs, sem_ids_path)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _generate_semantic_id(self, vqvae_model: MultiHeadVQVAE,
                              sent_embs: torch.Tensor,
                              sem_ids_path: str) -> None:
        vqvae_model.eval()

        batch_size = self.config['vqvae_batch_size']
        data = EmbDataset(sent_embs.cpu())
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True)

        all_indices = []
        all_indices_str = []

        for d in tqdm(dataloader):
            d = d.to(self.config['device'])

            indices = vqvae_model.get_indices(d, use_sk=False)

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = index.tolist()

                all_indices.append(code)
                all_indices_str.append('-'.join([str(x) for x in code]))

        all_indices = np.array(all_indices)
        all_indices_str = np.array(all_indices_str)

        use_sk = (self.config['vqvae_sk_epsilon']
                  > 0) and (self.config['vqvae_sk_iters'] > 0)
        
        if use_sk:
            for _ in range(30):
                # for _ in range(1):
                if self._check_collision(all_indices_str):
                    break

                collision_item_groups = self._get_collision_items(
                    all_indices_str)

                for collision_items in collision_item_groups:
                    d = data[collision_items].to(self.config['device'])
                    indices = vqvae_model.get_indices(d, use_sk=True)

                    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                    for item, index in zip(collision_items, indices):
                        code = index.tolist()

                        all_indices[item] = code
                        all_indices_str[item] = '-'.join(
                            [str(x) for x in code])

        self.log(f"All indices number: {len(all_indices)}")

        indices_count = defaultdict(int)
        for index in all_indices_str:
            indices_count[index] += 1

        # self.log("Max number of conflicts: ", max(indices_count.values()))
        self.log(f"Max number of conflicts: {max(indices_count.values())}")

        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        self.log("Collision Rate :{}".format(
            (tot_item - tot_indice) / tot_item))

        item2sem_ids = {}
        for item_id, indices in enumerate(all_indices.tolist()):
            item = self.id2item[int(item_id) + 1]
            item2sem_ids[item] = list(indices)

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            # json.dump(all_indices_str, f)
            json.dump(item2sem_ids, f)

    def _get_collision_items(self, str_sem_ids):
        sem_id2item = defaultdict(list)
        for i, str_sem_id in enumerate(str_sem_ids):
            sem_id2item[str_sem_id].append(i)

        collision_item_groups = []
        for str_sem_id in sem_id2item:
            if len(sem_id2item[str_sem_id]) > 1:
                collision_item_groups.append(sem_id2item[str_sem_id])

        return collision_item_groups

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] +
                                  self.codebook_sizes[digit - 1])
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids
