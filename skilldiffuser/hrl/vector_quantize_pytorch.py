import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, repeat
from contextlib import contextmanager


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# regularization losses


def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)

# distance types


def info_nce(code, embeds, pos_indices, dist):
    # assume embeds are B, T, D
    # codes are B, T, D with N quantized values

    B, T, D = code.shape
    # calculate dist matrix B, T, N
    dist = dist.reshape(B, T, -1)
    # might need to accomodate time

    # print(code.shape, embeds.shape, dist.shape)

    # pos_loss = 1/2 * (code - embeds).pow(2).sum(dim=-1, keepdim=True)  # B, T, 1
    # contrast_loss = torch.logsumexp(dist/2, dim=0, keepdim=True)  # sum over batch dim to generate negatie samples
    # loss = pos_loss + contrast_loss
    p = torch.log_softmax(dist/2, dim=-1)
    # loss = -torch.log_softmax(p, dim=0)
    loss = -p
    # pos_mask = dist.max(dim=-1).indices
    loss = loss[pos_indices]

    # print(pos_loss.shape, contrast_loss.shape)
    return loss.mean()


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.t()

        self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        return quantize, embed_ind, dist


class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(torch.randn(codebook_size, dim))
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters,
                                     use_cosine_sim=True)
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        samples = l2norm(samples)
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = l2norm(self.embed)
        dist = flatten @ embed.t()
        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(0)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], embed,
                                           embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        return quantize, embed_ind, dist

# main class


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed=None,
        codebook_dim=None,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        use_cosine_sim=False,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=None,
        commitment=1.,  # deprecate in next version, turn off by default
        orthogonal_reg_weight=0.,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        quantize_full_precision=False,
        cpc=False,
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection \
            else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection \
            else nn.Identity()

        self.eps = eps
        self.commitment_weight = default(commitment_weight, commitment)

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim \
            else CosineSimCodebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            codebook_size=n_embed,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

        # when quantizing, whether to turn off mixed precision
        self.quantize_full_precision = quantize_full_precision

        # use contrastive-predictive-coding for commitment loss
        self.cpc = cpc

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x):
        shape, device, codebook_size = x.shape, x.device, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        context = null_context() if not self.quantize_full_precision else autocast(enabled=False)

        with context:
            quantize, embed_ind, dist = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.cpc:
                    commit_loss = info_nce(quantize.detach(), x, embed_ind, dist)
                else:
                    commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) -> b h w', h=height, w=width)

        return quantize, embed_ind, loss
