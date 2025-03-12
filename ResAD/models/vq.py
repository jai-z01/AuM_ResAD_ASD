from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor, m: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)  # (N, dim)
        m = m.reshape(-1)
        z_normal = z_flattened[m == 0]

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_normal, self.embedding.weight), dim=1)  # (N, )

        z_q = self.embedding(min_encoding_indices)
        perplexity = None
        min_encodings = None

        loss = torch.mean((z_q.detach() - z_normal) ** 2) + self.beta * torch.mean((z_q - z_normal.detach()) ** 2)

        # preserve gradients
        z_q: torch.FloatTensor = z_normal + (z_q - z_normal).detach()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_condebook_entry(self, z: torch.FloatTensor) -> torch.FloatTensor:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)  # (N, dim)

        # Process cdist in chunks
        chunk_size = 1024  # Adjust based on available memory
        min_encoding_indices = []
        
        for i in range(0, z_flattened.size(0), chunk_size):
            chunk = z_flattened[i : i + chunk_size]
            distances = torch.cdist(chunk, self.embedding.weight)
            min_encoding_indices.append(torch.argmin(distances, dim=1))
        
        min_encoding_indices = torch.cat(min_encoding_indices, dim=0)  # Merge results

        z_q = self.embedding(min_encoding_indices).view(z.shape)  # (batch, height, width, channel)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class MultiScaleVQ(nn.Module):
    def __init__(self, 
                 num_embeddings = 1024,
                 channels = (256, 512, 1024)):
        super().__init__()
        self.vq1 = VectorQuantizer(num_embeddings, channels[0], beta=0.25, remap=None, sane_index_shape=False)
        self.vq2 = VectorQuantizer(num_embeddings, channels[1], beta=0.25, remap=None, sane_index_shape=False)
        self.vq3 = VectorQuantizer(num_embeddings, channels[2], beta=0.25, remap=None, sane_index_shape=False)
    
    def forward(self, features, masks=None, train=True):
        if train:
            _, loss1, _ = self.vq1(features[0], masks[0])
            _, loss2, _ = self.vq2(features[1], masks[1])
            _, loss3, _ = self.vq3(features[2], masks[2])
            loss = loss1 + loss2 + loss3
            return loss
        else:
            qx1 = self.vq1.get_condebook_entry(features[0])
            qx2 = self.vq2.get_condebook_entry(features[1])
            qx3 = self.vq3.get_condebook_entry(features[2])
            return qx1, qx2, qx3


class MultiScaleVQ4(nn.Module):
    def __init__(self, 
                 num_embeddings = 1024,
                 channels = (1280, 1280, 1280, 1280)):
        super().__init__()
        self.vq1 = VectorQuantizer(num_embeddings, channels[0], beta=0.25, remap=None, sane_index_shape=False)
        self.vq2 = VectorQuantizer(num_embeddings, channels[1], beta=0.25, remap=None, sane_index_shape=False)
        self.vq3 = VectorQuantizer(num_embeddings, channels[2], beta=0.25, remap=None, sane_index_shape=False)
        self.vq4 = VectorQuantizer(num_embeddings, channels[3], beta=0.25, remap=None, sane_index_shape=False)
    
    def forward(self, features, masks=None, train=True):
        if train:
            _, loss1, _ = self.vq1(features[0], masks[0])
            _, loss2, _ = self.vq2(features[1], masks[1])
            _, loss3, _ = self.vq3(features[2], masks[2])
            _, loss4, _ = self.vq4(features[3], masks[3])
            loss = loss1 + loss2 + loss3 + loss4
            return loss
        else:
            qx1 = self.vq1.get_condebook_entry(features[0])
            qx2 = self.vq2.get_condebook_entry(features[1])
            qx3 = self.vq3.get_condebook_entry(features[2])
            qx4 = self.vq4.get_condebook_entry(features[3])
            return qx1, qx2, qx3, qx4