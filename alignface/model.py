from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

ATTRIBUTES = [
    "eyebrow_shape",
    "eyebrow_thickness",
    "eye_shape",
    "eye_size",
    "eye_colour",
    "nose_shape",
    "nose_size",
    "mouth_size",
    "lip_thickness",
    "lip_colour",
    "face_proportion",
    "forehead_height",
    "pupillary_distance",
    "cheek_shape",
    "jaw_width",
    "chin_shape",
    "hair_colour",
    "hair_length",
    "skin_colour",
    "skin_texture",
]

ATTRIBUTE_TEXT: Dict[str, str] = {
    "eyebrow_shape": "eyebrow shape",
    "eyebrow_thickness": "eyebrow thickness",
    "eye_shape": "eye shape",
    "eye_size": "eye size",
    "eye_colour": "eye colour",
    "nose_shape": "nose shape",
    "nose_size": "nose size",
    "mouth_size": "mouth size",
    "lip_thickness": "lip thickness",
    "lip_colour": "lip colour",
    "face_proportion": "facial proportions",
    "forehead_height": "forehead height",
    "pupillary_distance": "distance between the pupils",
    "cheek_shape": "cheek shape",
    "jaw_width": "jaw width",
    "chin_shape": "chin shape",
    "hair_colour": "hair colour",
    "hair_length": "hair length",
    "skin_colour": "skin colour",
    "skin_texture": "skin texture",
}

def get_attr_text_list(selected_attributes: List[str]) -> List[str]:
    texts = []
    for attr in selected_attributes:
        if attr not in ATTRIBUTE_TEXT:
            raise ValueError(f"Missing ATTRIBUTE_TEXT for {attr}")
        texts.append(ATTRIBUTE_TEXT[attr])
    return texts

def _tree_interp_1d(x: torch.Tensor, edges: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Piecewise linear interpolation: x [B], edges [n+1], values [n+1].
    Returns [B] interpolated values.
    """
    x_min, x_max = edges[0], edges[-1]
    span = (x_max - x_min).clamp(min=1e-8)
    t = ((x - x_min) / span * (values.shape[0] - 1)).clamp(0.0, values.shape[0] - 1 - 1e-6)
    left = t.long().clamp(0, values.shape[0] - 2)
    right = left + 1
    frac = t - left.float()
    v_left = values[left]
    v_right = values[right]
    return (1 - frac) * v_left + frac * v_right

class Stage2MonotonicGAMHead(nn.Module):
    """
    GAM with monotonic non-decreasing main effects along x per dimension.

    For each attribute d, knot values v_d[0..n_bins] satisfy v_d[k+1] >= v_d[k] by
    v_d[0] = main_base[d], v_d[k+1] = v_d[k] + softplus(raw_deltas[d, k]).

    Same [x_min, x_max] and n_bins (number of bins / segments) as Stage2GAMHead; no interaction terms.

    **Forward (default ``per_dim_tanh_mean=True``):** u_d(x) interpolated from knots, t_d=tanh(u_d),
    δ = (bias + Σ_d t_d) / D when ``per_dim_tanh_divide_by_dim=True`` (default), or
    δ = bias + Σ_d t_d when ``per_dim_tanh_divide_by_dim=False``.
    If ``per_dim_tanh_mean=False`` (legacy checkpoints), optionally
    δ = tanh(bias + Σ u_d) when ``delta_tanh=True``, else raw sum.

    If ``gam_add_mean_term=True``, the pre-link linear index includes an extra copy of the
    mean term: ``S = Σ_d u_d`` becomes ``S + S/D`` (or ``Σ t_d + (Σ t_d)/D`` in the tanh-per-dim
    branch). For ``sigmoid_signed`` stage-2 training this scales logits by ``(1 + 1/D)`` so each
    marginal can grow larger before saturation.

    Note: this head assumes input x is already transformed onto the same axis as ``edges``.
    In the aligned stage-2 pipeline we use stage1 x_raw -> per-dim minmax to [0,1], and
    train this head with x_min=0, x_max=1.
    """

    def __init__(
        self,
        input_dim: int,
        n_bins: int = 50,
        x_min: float = 0.0,
        x_max: float = 1.0,
        dropout: float = 0.5,
        delta_tanh: bool = True,
        per_dim_tanh_mean: bool = True,
        per_dim_tanh_divide_by_dim: bool = True,
        use_bias: bool = True,
        gam_add_mean_term: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_bins = n_bins
        self.x_min = x_min
        self.x_max = x_max

        self.delta_tanh = bool(delta_tanh)

        self.per_dim_tanh_mean = bool(per_dim_tanh_mean)

        self.per_dim_tanh_divide_by_dim = bool(per_dim_tanh_divide_by_dim)
        self.use_bias = bool(use_bias)

        self.gam_add_mean_term = bool(gam_add_mean_term)

        edges = torch.linspace(x_min, x_max, n_bins + 1)
        self.register_buffer("edges", edges)

        self.main_base = nn.Parameter(torch.zeros(input_dim))
        init_raw = -3.0
        raw_init = init_raw + 0.4 * (torch.rand(input_dim, n_bins) - 0.5)
        self.main_raw_deltas = nn.Parameter(raw_init)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("bias", torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def _main_leaf_values(self) -> torch.Tensor:




        sp = F.softplus(self.main_raw_deltas)
        gate = torch.sigmoid((self.main_raw_deltas + 2.9) * 10.0)
        incr = sp * gate
        cum = torch.cumsum(incr, dim=-1)
        return torch.cat(
            [self.main_base.unsqueeze(-1), self.main_base.unsqueeze(-1) + cum],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        edges = self.edges
        vals = self._main_leaf_values()
        d_in = self.input_dim
        if self.per_dim_tanh_mean:
            terms = []
            for d in range(d_in):
                u_d = _tree_interp_1d(x[:, d], edges, vals[d])
                terms.append(torch.tanh(u_d))
            tsum = torch.stack(terms, dim=1).sum(dim=1)
            if self.gam_add_mean_term:
                tsum = tsum + tsum / float(d_in)
            out = self.bias.squeeze(0) + tsum
            if self.per_dim_tanh_divide_by_dim:
                out = out / float(d_in)
            return out

        out = x.new_zeros(x.shape[0])
        for d in range(d_in):
            out = out + _tree_interp_1d(x[:, d], edges, vals[d])
        if self.gam_add_mean_term:
            out = out + out / float(d_in)
        raw = out + self.bias.squeeze(0)
        if self.delta_tanh:
            return torch.tanh(raw)
        return raw

    def forward_components(self, x: torch.Tensor):
        """Return (main_term, 0): sum of raw u_d (no bias); same as pre-tanh path when not per_dim_tanh_mean."""
        x = self.dropout(x)
        edges = self.edges
        vals = self._main_leaf_values()
        main_term = x.new_zeros(x.shape[0])
        for d in range(self.input_dim):
            main_term = main_term + _tree_interp_1d(x[:, d], edges, vals[d])
        if self.gam_add_mean_term:
            main_term = main_term + main_term / float(self.input_dim)
        return main_term, x.new_zeros(x.shape[0])

class Stage2MonotonicGAMAttrSoftmaxHead(nn.Module):
    """
    Global attribute weights on monotonic main effects:

        a = softmax(logits) in R^D,  sum_i a_i = 1,  a_i > 0,
        s(x) = sum_i a_i f_i(x_i).

    ``logits`` are learnable (one per attribute); not input-dependent.
    The f_i branch matches ``Stage2MonotonicGAMHead`` norm settings (raw u_i, no bias).
    """

    def __init__(
        self,
        input_dim: int,
        n_bins: int = 50,
        x_min: float = -1.0,
        x_max: float = 1.0,
        gam_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.gam = Stage2MonotonicGAMHead(
            input_dim=input_dim,
            n_bins=n_bins,
            x_min=x_min,
            x_max=x_max,
            dropout=gam_dropout,
            delta_tanh=False,
            per_dim_tanh_mean=False,
            per_dim_tanh_divide_by_dim=False,
            use_bias=False,
            gam_add_mean_term=False,
        )
        self.attr_logits = nn.Parameter(torch.zeros(self.input_dim))

    @property
    def delta_tanh(self) -> bool:
        return self.gam.delta_tanh

    @property
    def per_dim_tanh_mean(self) -> bool:
        return self.gam.per_dim_tanh_mean

    @property
    def per_dim_tanh_divide_by_dim(self) -> bool:
        return self.gam.per_dim_tanh_divide_by_dim

    @property
    def use_bias(self) -> bool:
        return self.gam.use_bias

    @property
    def gam_add_mean_term(self) -> bool:
        return self.gam.gam_add_mean_term

    def attr_weights(self) -> torch.Tensor:
        """(D,) positive, sum to 1."""
        return F.softmax(self.attr_logits, dim=0)

    def per_dim_u(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) raw f_i / u_i from monotonic splines."""
        edges = self.gam.edges
        vals = self.gam._main_leaf_values()
        parts = []
        for d in range(self.input_dim):
            parts.append(_tree_interp_1d(x[:, d], edges, vals[d]).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def forward_detailed(self, x: torch.Tensor):
        """
        Returns (s, u, w, logits_broadcast).
        w is (B, D) with the same global a_i on every row (for c_i = w_i u_i plots).
        logits_broadcast is (B, D) copy of attr_logits for debugging.
        """
        x = self.gam.dropout(x)
        u = self.per_dim_u(x)
        a = self.attr_weights()
        w = a.unsqueeze(0).expand_as(u)
        s = (u * w).sum(dim=-1)
        logits_b = self.attr_logits.unsqueeze(0).expand_as(u)
        return s, u, w, logits_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_detailed(x)[0]

class LoRALinear(nn.Module):
    """Drop-in linear layer with additive low-rank adaptation."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.base = base
        self.dropout = nn.Dropout(lora_dropout)

        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)


        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_d = self.dropout(x)

        lora_out = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """Return (parent_module, leaf_name) for a dotted module path."""
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def apply_lora_to_clip_vision(
    clip_model: CLIPModel,
    rank: int = 4,
    alpha: float = 8.0,
    lora_dropout: float = 0.0,
    target_substrings: Optional[Iterable[str]] = None,
) -> int:
    """
    Replace selected nn.Linear in clip_model.vision_model with LoRALinear.
    Returns number of wrapped layers.
    """
    if target_substrings is None:
        target_substrings = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    target_substrings = tuple(target_substrings)

    to_wrap: List[str] = []
    for name, module in clip_model.vision_model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_substrings):
            to_wrap.append(name)

    wrapped = 0
    for name in to_wrap:
        parent, leaf = _get_parent_module(clip_model.vision_model, name)
        base_linear = getattr(parent, leaf)
        if not isinstance(base_linear, nn.Linear):
            continue
        setattr(
            parent,
            leaf,
            LoRALinear(
                base=base_linear,
                rank=rank,
                alpha=alpha,
                lora_dropout=lora_dropout,
            ),
        )
        wrapped += 1
    return wrapped

class CrossAttentionAttributeHead(nn.Module):
    """
    Cross-attention fusion head (same as model_v2_crossattn).
    - query: pair embedding (projected to attn_dim)
    - key/value: text side (concat of learnable prefix tokens + CLIP prompt emb, projected to attn_dim)
    Fusion: fused = pair_token + tanh(gate) * proj(cross_attn(...))
    """

    def __init__(
        self,
        pair_dim: int,
        text_in_dim: int,
        attn_dim: int = 128,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.text_in_dim = int(text_in_dim)
        self.attn_dim = int(attn_dim)
        if self.attn_dim <= 0:
            raise ValueError(f"attn_dim must be > 0, got {self.attn_dim}")
        if self.attn_dim % int(num_heads) != 0:
            raise ValueError(
                f"attn_dim ({self.attn_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.pair_to_attn = nn.Linear(self.pair_dim, self.attn_dim)
        self.text_to_attn = nn.Linear(self.text_in_dim, self.attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_to_pair = nn.Linear(self.attn_dim, self.pair_dim)
        self.attn_gate = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.fused_mlp = nn.Sequential(
            nn.Linear(self.pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_feat: torch.Tensor, text_feat: torch.Tensor):

        b, n, _ = pair_feat.shape
        pair_token = pair_feat.reshape(b * n, 1, -1)
        text_token = text_feat.reshape(b * n, 1, -1)
        pair_token_attn = self.pair_to_attn(pair_token)
        text_token_attn = self.text_to_attn(text_token)
        attn_out_attn, _ = self.cross_attn(
            query=pair_token_attn,
            key=text_token_attn,
            value=text_token_attn,
            need_weights=False,
        )
        attn_out = self.attn_to_pair(attn_out_attn)
        fused = (pair_token + torch.tanh(self.attn_gate) * attn_out).reshape(b, n, -1)

        logits_fused = self.fused_mlp(fused).squeeze(-1)
        logits = logits_fused
        sim = torch.sigmoid(logits)
        return sim, logits

class CLIPSharedAttributeModel(nn.Module):
    """
    V2 stage-1 model:
    - Same forward behavior as model.CLIPSharedAttributeModel
    - Optional LoRA on image encoder vision branch.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        sim_norm_momentum: float = 0.05,
        sim_norm_eps: float = 1e-6,
        sim_norm_scale: float = 3.0,
        num_attributes: int = len(ATTRIBUTES),
        num_learnable_text_prefix_tokens: int = 2,
        cross_attn_dim: int = 128,
        cross_attn_heads: int = 2,
        backbone_config=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = (
            list(lora_target_modules)
            if lora_target_modules is not None
            else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        )
        if not (0.0 < float(sim_norm_momentum) <= 1.0):
            raise ValueError(f"sim_norm_momentum must be in (0, 1], got {sim_norm_momentum}")
        if float(sim_norm_eps) <= 0.0:
            raise ValueError(f"sim_norm_eps must be > 0, got {sim_norm_eps}")
        if float(sim_norm_scale) <= 0.0:
            raise ValueError(f"sim_norm_scale must be > 0, got {sim_norm_scale}")
        self.sim_norm_momentum = float(sim_norm_momentum)
        self.sim_norm_eps = float(sim_norm_eps)
        self.sim_norm_scale = float(sim_norm_scale)
        self.num_attributes = int(num_attributes)
        self.num_learnable_text_prefix_tokens = int(num_learnable_text_prefix_tokens)
        if self.num_attributes <= 0:
            raise ValueError(f"num_attributes must be > 0, got {self.num_attributes}")
        if self.num_learnable_text_prefix_tokens < 0:
            raise ValueError(
                f"num_learnable_text_prefix_tokens must be >= 0, got {self.num_learnable_text_prefix_tokens}"
            )
        self.clip = CLIPModel(backbone_config) if backbone_config is not None else CLIPModel.from_pretrained(model_name)

        image_dim = self.clip.config.projection_dim
        text_dim = self.clip.config.projection_dim
        pair_dim = image_dim * 5
        self.text_learnable_tokens = nn.Parameter(
            torch.randn(self.num_attributes, self.num_learnable_text_prefix_tokens, text_dim) * 0.02
        )
        text_in_dim = (1 + self.num_learnable_text_prefix_tokens) * text_dim
        self.head = CrossAttentionAttributeHead(
            pair_dim=pair_dim,
            text_in_dim=text_in_dim,
            attn_dim=int(cross_attn_dim),
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=int(cross_attn_heads),
        )

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.num_lora_wrapped = 0
        if self.use_lora:
            self.num_lora_wrapped = apply_lora_to_clip_vision(
                self.clip,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_substrings=self.lora_target_modules,
            )
        n_attr_max = len(ATTRIBUTES)
        self.register_buffer("sim_running_mean", torch.zeros(n_attr_max, dtype=torch.float32))
        self.register_buffer("sim_running_var", torch.ones(n_attr_max, dtype=torch.float32))
        self.register_buffer("sim_stats_initialized", torch.tensor(False, dtype=torch.bool))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        feat = self.clip.visual_projection(pooled_output)
        return F.normalize(feat, p=2, dim=1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs.pooler_output
        feat = self.clip.text_projection(pooled_output)
        return F.normalize(feat, p=2, dim=1)

    @staticmethod
    def build_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        abs_diff = torch.abs(diff)
        mul = x * y
        return torch.cat([x, y, diff, abs_diff, mul], dim=-1)

    def _update_running_stats(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, n_attr: int) -> None:
        with torch.no_grad():
            if bool(self.sim_stats_initialized.item()):
                m = self.sim_norm_momentum
                self.sim_running_mean[:n_attr].mul_(1.0 - m).add_(m * batch_mean)
                self.sim_running_var[:n_attr].mul_(1.0 - m).add_(m * batch_var)
            else:
                self.sim_running_mean[:n_attr].copy_(batch_mean)
                self.sim_running_var[:n_attr].copy_(batch_var)
                self.sim_stats_initialized.fill_(True)

    def _normalized_sim_from_logits(
        self,
        logits_ar: torch.Tensor,
        logits_br: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize logits with z-score per attribute, then map to (0, 1) by sigmoid.
        - train(): update running mean/var from current batch and use batch stats.
        - eval(): use fixed running mean/var.
        """
        n_attr = logits_ar.size(1)
        if n_attr > self.sim_running_mean.numel():
            raise ValueError(f"n_attr={n_attr} exceeds stats capacity={self.sim_running_mean.numel()}")

        if self.training:
            both = torch.cat([logits_ar.detach(), logits_br.detach()], dim=0)
            mean = both.mean(dim=0)
            var = both.var(dim=0, unbiased=False)
            self._update_running_stats(mean, var, n_attr=n_attr)
        else:
            if bool(self.sim_stats_initialized.item()):
                mean = self.sim_running_mean[:n_attr]
                var = self.sim_running_var[:n_attr]
            else:
                both = torch.cat([logits_ar.detach(), logits_br.detach()], dim=0)
                mean = both.mean(dim=0)
                var = both.var(dim=0, unbiased=False)

        mean = mean.to(device=logits_ar.device, dtype=logits_ar.dtype)
        var = var.to(device=logits_ar.device, dtype=logits_ar.dtype)
        denom = torch.sqrt(var + self.sim_norm_eps)
        z_ar = (logits_ar - mean) / denom
        z_br = (logits_br - mean) / denom
        sim_ar = torch.sigmoid(self.sim_norm_scale * z_ar)
        sim_br = torch.sigmoid(self.sim_norm_scale * z_br)
        return sim_ar, sim_br

    def forward(
        self,
        ref: torch.Tensor,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        attr_text_embs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        emb_r = self.encode_image(ref)
        emb_a = self.encode_image(img_a)
        emb_b = self.encode_image(img_b)

        pair_ar = self.build_pair(emb_a, emb_r)
        pair_br = self.build_pair(emb_b, emb_r)

        b = ref.size(0)
        n_attr = attr_text_embs.size(0)
        if n_attr != self.num_attributes:
            raise ValueError(
                f"attr_text_embs has n_attr={n_attr} but model.num_attributes={self.num_attributes}"
            )
        pair_ar = pair_ar.unsqueeze(1).expand(b, n_attr, -1)
        pair_br = pair_br.unsqueeze(1).expand(b, n_attr, -1)
        lt = self.text_learnable_tokens.unsqueeze(0).expand(b, n_attr, -1, -1)
        prompt = attr_text_embs.unsqueeze(0).expand(b, n_attr, -1).unsqueeze(2)
        text_feat = torch.cat([lt, prompt], dim=2).reshape(b, n_attr, -1)

        _, logits_ar = self.head(pair_ar, text_feat)
        _, logits_br = self.head(pair_br, text_feat)
        sim_ar, sim_br = self._normalized_sim_from_logits(logits_ar, logits_br)
        delta_logit = logits_ar - logits_br
        return {
            "sim_ar": sim_ar,
            "sim_br": sim_br,
            "logits_ar": logits_ar,
            "logits_br": logits_br,
            "delta_sim": sim_ar - sim_br,
            "delta_logit": delta_logit,
        }
