import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


class FixedCoverageMaxTransitionsBatchSampler(torch.utils.data.BatchSampler):

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        resolved_transitions_by_local,
        *,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.resolved_transitions_by_local = [
            int(v) for v in resolved_transitions_by_local
        ]
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if len(self.resolved_transitions_by_local) != self.num_samples:
            raise ValueError(
                "resolved_transitions_by_local length must equal num_samples. "
                f"got {len(self.resolved_transitions_by_local)} vs {self.num_samples}."
            )
        if any(v < 1 for v in self.resolved_transitions_by_local):
            raise ValueError(
                "resolved_transitions_by_local must all be >= 1, "
                f"got min={min(self.resolved_transitions_by_local)}."
            )

    @staticmethod
    def _dist_info():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rank, world_size = self._dist_info()

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        ordered = torch.randperm(self.num_samples, generator=generator).tolist()

        if world_size > 1:
            if self.drop_last:
                total_usable = (
                    self.num_samples // (world_size * self.batch_size)
                ) * (world_size * self.batch_size)
                ordered = ordered[:total_usable]
                total_size = len(ordered)
            else:
                per_rank = math.ceil(self.num_samples / world_size)
                total_size = per_rank * world_size
                if total_size > len(ordered):
                    ordered = ordered + ordered[: total_size - len(ordered)]

            ordered = ordered[rank:total_size:world_size]

        buckets = defaultdict(list)
        for local_idx in ordered:
            transition_len = self.resolved_transitions_by_local[local_idx]
            buckets[int(transition_len)].append(local_idx)

        batched = []
        for transition_len, bucket_indices in buckets.items():
            perm = torch.randperm(len(bucket_indices), generator=generator).tolist()
            ordered_bucket = [bucket_indices[p] for p in perm]

            if self.drop_last:
                usable = (len(ordered_bucket) // self.batch_size) * self.batch_size
                ordered_bucket = ordered_bucket[:usable]

            for i in range(0, len(ordered_bucket), self.batch_size):
                batch_indices = ordered_bucket[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batched.append(
                    [(local_idx, int(transition_len)) for local_idx in batch_indices]
                )

        if len(batched) > 0:
            order = torch.randperm(len(batched), generator=generator).tolist()
            for i in order:
                yield batched[i]

    def __len__(self):
        rank, world_size = self._dist_info()
        del rank

        if world_size == 1:
            buckets = self._buckets_for_len_estimation()
            if self.drop_last:
                return sum(len(bucket) // self.batch_size for bucket in buckets.values())
            return sum(math.ceil(len(bucket) / self.batch_size) for bucket in buckets.values())

        if self.drop_last:
            total_usable = (
                self.num_samples // (world_size * self.batch_size)
            ) * (world_size * self.batch_size)
            per_rank_samples = total_usable // world_size
            return per_rank_samples // self.batch_size

        per_rank_samples = math.ceil(self.num_samples / world_size)
        return math.ceil(per_rank_samples / self.batch_size)

    def _buckets_for_len_estimation(self):
        buckets = defaultdict(list)
        for local_idx, transition_len in enumerate(self.resolved_transitions_by_local):
            buckets[int(transition_len)].append(local_idx)
        return buckets


def _episode_column_name(dataset) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    return ""


def _resolve_alignment_mode(dataset) -> str:
    """
    Prefer episode+step alignment when available.
    Fall back to global-index alignment otherwise.
    """
    episode_col = _episode_column_name(dataset)
    has_step = "step_idx" in dataset.column_names
    if episode_col and has_step:
        return "episode_step"
    return "global_index"


def _build_key_to_row_index(dataset, mode: str, episode_col: str = ""):
    """
    Build key -> row index map for a dataset.

    mode == "episode_step": key = (episode_idx, step_idx)
    mode == "global_index": key = (0, global_row_idx)
    """
    if mode == "episode_step":
        episode_idx = dataset.get_col_data(episode_col)
        step_idx = dataset.get_col_data("step_idx")
        if len(episode_idx) != len(step_idx):
            raise ValueError("Mismatched lengths for episode and step columns.")

        mapping = {}
        for i in range(len(step_idx)):
            key = (int(episode_idx[i]), int(step_idx[i]))
            if key in mapping:
                raise ValueError(f"Duplicate alignment key found in dataset: {key}")
            mapping[key] = int(i)
        return mapping

    if mode == "global_index":
        return {(0, int(i)): int(i) for i in range(len(dataset))}

    raise ValueError(f"Unknown alignment mode: {mode}")


class FixedStartVariableTransitionsDataset(torch.utils.data.Dataset):
    """
    Dataset indexed by (local_train_idx, effective_transitions).

    - local_train_idx indexes the original base train starts.
    - effective_transitions is resolved by sampler.
    """

    def __init__(
        self,
        datasets_by_transitions: dict[int, torch.utils.data.Dataset],
        base_keys,
        fallback_transitions: int,
    ):
        if len(datasets_by_transitions) == 0:
            raise ValueError("datasets_by_transitions must be non-empty.")
        if len(base_keys) == 0:
            raise ValueError("base_keys must be non-empty.")

        self.datasets_by_transitions = {
            int(k): v for k, v in datasets_by_transitions.items()
        }
        self.supported_transitions = sorted(self.datasets_by_transitions.keys())
        self.fallback_transitions = int(fallback_transitions)
        if self.fallback_transitions not in self.datasets_by_transitions:
            raise ValueError(
                f"fallback_transitions={self.fallback_transitions} is not in "
                f"datasets_by_transitions {self.supported_transitions}."
            )

        self.base_keys = [(int(a), int(b)) for a, b in base_keys]

        reference_dataset = self.datasets_by_transitions[self.fallback_transitions]
        self.alignment_mode = _resolve_alignment_mode(reference_dataset)
        self.episode_col = _episode_column_name(reference_dataset)

        self.key_to_row_by_transitions = {}
        for transitions, dataset in self.datasets_by_transitions.items():
            cur_mode = _resolve_alignment_mode(dataset)
            if cur_mode != self.alignment_mode:
                raise ValueError(
                    f"Inconsistent alignment mode for transitions={transitions}: "
                    f"{cur_mode} vs {self.alignment_mode}."
                )

            cur_episode_col = _episode_column_name(dataset)
            if self.alignment_mode == "episode_step" and cur_episode_col != self.episode_col:
                raise ValueError(
                    f"Inconsistent episode column for transitions={transitions}: "
                    f"{cur_episode_col} vs {self.episode_col}."
                )

            self.key_to_row_by_transitions[transitions] = _build_key_to_row_index(
                dataset,
                mode=self.alignment_mode,
                episode_col=self.episode_col,
            )

        fallback_map = self.key_to_row_by_transitions[self.fallback_transitions]
        missing_in_fallback = [k for k in self.base_keys if k not in fallback_map]
        if missing_in_fallback:
            preview = ", ".join(str(k) for k in missing_in_fallback[:5])
            raise ValueError(
                f"fallback_transitions={self.fallback_transitions} misses "
                f"{len(missing_in_fallback)} base keys. Examples: {preview}"
            )

        self.valid_transitions_by_local = []
        for key in self.base_keys:
            valid_t = {
                t
                for t, key_to_row in self.key_to_row_by_transitions.items()
                if key in key_to_row
            }
            if self.fallback_transitions not in valid_t:
                raise ValueError(
                    f"Base key {key} is not valid at fallback_transitions="
                    f"{self.fallback_transitions}."
                )
            self.valid_transitions_by_local.append(valid_t)

    def __len__(self):
        return len(self.base_keys)

    def resolve_local_transitions(
        self, local_idx: int, requested_max_transitions: int
    ) -> int:
        local_idx = int(local_idx)
        requested_max_transitions = int(requested_max_transitions)
        if local_idx < 0 or local_idx >= len(self.base_keys):
            raise IndexError(f"local_idx={local_idx} out of range [0, {len(self.base_keys)}).")
        valid = self.valid_transitions_by_local[local_idx]
        eligible = [t for t in valid if t <= requested_max_transitions]
        if len(eligible) > 0:
            return int(max(eligible))
        return int(min(valid))

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "FixedStartVariableTransitionsDataset expects keys of shape "
                "(local_train_idx, effective_transitions)."
            )

        local_idx = int(key[0])
        effective_transitions = int(key[1])

        if local_idx < 0 or local_idx >= len(self.base_keys):
            raise IndexError(f"local_idx={local_idx} out of range [0, {len(self.base_keys)}).")
        if effective_transitions not in self.datasets_by_transitions:
            raise KeyError(
                f"Unknown effective_transitions={effective_transitions}, "
                f"choices={self.supported_transitions}."
            )

        sample_key = self.base_keys[local_idx]
        row_idx = self.key_to_row_by_transitions[effective_transitions][sample_key]
        sample = self.datasets_by_transitions[effective_transitions][row_idx]

        if isinstance(sample, dict):
            sample = dict(sample)
            sample["dense_transitions"] = torch.tensor(
                effective_transitions, dtype=torch.long
            )
        return sample


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    lambd = cfg.loss.sigreg.weight
    dense_cfg = cfg.wm.get("dense_supervision", None)

    supervise_last_state_only = bool(
        dense_cfg and dense_cfg.get("supervise_last_state_only", False)
    )
    action_frame_skip = int(cfg.wm.get("action_frame_skip", 1))
    if action_frame_skip < 1:
        raise ValueError(f"wm.action_frame_skip must be >= 1, got {action_frame_skip}.")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    model_batch = dict(batch)

    if action_frame_skip > 1:
        action = model_batch["action"]
        pixels = model_batch["pixels"]

        raw_transitions = min(int(action.size(1)), int(pixels.size(1) - 1))
        packed_transitions = raw_transitions // action_frame_skip
        if packed_transitions < 1:
            raise ValueError(
                "Not enough transitions to build one frame-skip action token. "
                f"raw_transitions={raw_transitions}, action_frame_skip={action_frame_skip}."
            )

        usable_raw_transitions = packed_transitions * action_frame_skip
        action = action[:, :usable_raw_transitions]
        action = rearrange(
            action,
            "b (f k) a -> b f (k a)",
            f=packed_transitions,
            k=action_frame_skip,
        )
        pixels = pixels[:, : usable_raw_transitions + 1][:, ::action_frame_skip]

        model_batch["action"] = action
        model_batch["pixels"] = pixels

    action = model_batch["action"]
    pixels = model_batch["pixels"]
    available_prefix = min(int(action.size(1)), int(pixels.size(1) - 1))
    if available_prefix < 1:
        raise ValueError(
            "Need at least one prefix transition for supervision. "
            f"action_shape={tuple(action.shape)}, pixels_shape={tuple(pixels.shape)}."
        )

    min_transitions = int(dense_cfg.get("min_transitions", 1))
    max_transitions = int(dense_cfg.get("max_transitions", min_transitions))
    if min_transitions < 1 or max_transitions < min_transitions:
        raise ValueError(
            "Invalid wm.dense_supervision transition range: "
            f"min={min_transitions}, max={max_transitions}."
        )

    effective_max = min(max_transitions, available_prefix)
    if "dense_transitions" in batch:
        dense_transitions = batch["dense_transitions"]
        if not torch.is_tensor(dense_transitions):
            dense_transitions = torch.as_tensor(
                dense_transitions, device=action.device
            )
        dense_transitions = dense_transitions.to(
            device=action.device, dtype=torch.long
        )
        unique_transitions = torch.unique(dense_transitions)
        if unique_transitions.numel() != 1:
            raise ValueError(
                "Batch contains mixed dense_transitions, expected one value per batch. "
                f"unique={unique_transitions.tolist()}."
            )
        sampled_transitions = int(unique_transitions.item())
        sampled_transitions = max(1, min(sampled_transitions, effective_max))
    else:
        sampled_transitions = effective_max

    model_batch["action"] = action[:, :sampled_transitions]
    if supervise_last_state_only:
        model_batch["pixels"] = torch.stack(
            (pixels[:, 0], pixels[:, sampled_transitions]),
            dim=1,
        )

    output = self.model.encode(model_batch)

    emb = output["emb"]  # dense: (B, F+1, D); last-only: (B, 2, D)
    act_prefix = output["act_emb"]  # expected: (B, sampled_transitions, D)

    if supervise_last_state_only:
        ctx_act = act_prefix[:, sampled_transitions - 1 : sampled_transitions]
        tgt_emb = emb[:, -1:]
    else:
        ctx_act = act_prefix[:, :sampled_transitions]
        tgt_emb = emb[:, 1 : sampled_transitions + 1]

    # Fix start to dataset-defined start (step_idx), only vary end via sampled_transitions.
    ctx_emb = emb[:, 0]
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # expected: (B, F, D)

    output["pred_loss"] = self.model.prediction_loss(
        pred_emb,
        tgt_emb,
        supervise_last_state_only=supervise_last_state_only,
    )
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="Fast-lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################
    pl.seed_everything(int(cfg.seed), workers=True)

    dense_cfg = cfg.wm.get("dense_supervision", None)
    action_frame_skip = int(cfg.wm.get("action_frame_skip", 1))
    if action_frame_skip < 1:
        raise ValueError(
            f"wm.action_frame_skip must be >= 1, got {action_frame_skip}."
        )

    base_dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if "num_steps" in base_dataset_kwargs:
        configured_num_steps = int(base_dataset_kwargs["num_steps"])
        if configured_num_steps < 2:
            raise ValueError(
                f"data.dataset.num_steps must be >= 2, got {configured_num_steps}."
            )
        base_dataset_kwargs["num_steps"] = (
            (configured_num_steps - 1) * action_frame_skip + 1
        )
    if "keys_to_cache" in base_dataset_kwargs:
        # In max_adaptive mode, base_dataset is only used for stats/normalization.
        base_dataset_kwargs["keys_to_cache"] = []
    base_dataset = swm.data.HDF5Dataset(**base_dataset_kwargs, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(base_dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", base_dataset.get_dim(col))

        if hasattr(cfg.wm, "action_dim"):
            base_action_dim = int(cfg.wm.action_dim)
            cfg.wm.action_base_dim = base_action_dim
            cfg.wm.action_dim = int(base_action_dim * action_frame_skip)
        cfg.wm.action_frame_skip = int(action_frame_skip)

    transform = spt.data.transforms.Compose(*transforms)
    base_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))

    min_transitions = int(dense_cfg.get("min_transitions", 1))
    max_transitions = int(dense_cfg.get("max_transitions", min_transitions))

    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    reference_transitions = int(min_transitions)
    aux_keys_to_cache_cfg = dense_cfg.get("aux_keys_to_cache", [])
    aux_keys_to_cache = (
        None if aux_keys_to_cache_cfg is None else list(aux_keys_to_cache_cfg)
    )
    requested_transitions = list(range(min_transitions, max_transitions + 1))
    datasets_by_transitions = {}
    for transitions in requested_transitions:
        cur_kwargs = dict(dataset_kwargs)
        cur_kwargs["num_steps"] = int(transitions * action_frame_skip + 1)
        if (
            transitions != reference_transitions
            and aux_keys_to_cache is not None
            and "keys_to_cache" in cur_kwargs
        ):
            cur_kwargs["keys_to_cache"] = aux_keys_to_cache
        datasets_by_transitions[transitions] = swm.data.HDF5Dataset(
            **cur_kwargs,
            transform=transform,
        )

    reference_dataset = datasets_by_transitions[reference_transitions]
    train_subset, _ = spt.data.random_split(
        reference_dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    val_source_dataset = datasets_by_transitions[int(max_transitions)]
    val_gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    _, val_set = spt.data.random_split(
        val_source_dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=val_gen,
    )

    base_mode = _resolve_alignment_mode(reference_dataset)
    episode_col = _episode_column_name(reference_dataset)
    if base_mode == "episode_step":
        episode_idx = reference_dataset.get_col_data(episode_col)
        step_idx = reference_dataset.get_col_data("step_idx")
        base_keys = [
            (int(episode_idx[i]), int(step_idx[i]))
            for i in train_subset.indices
        ]
    else:
        base_keys = [(0, int(i)) for i in train_subset.indices]

    train_set = FixedStartVariableTransitionsDataset(
        datasets_by_transitions=datasets_by_transitions,
        base_keys=base_keys,
        fallback_transitions=reference_transitions,
    )

    resolved_transitions_by_local = [
        train_set.resolve_local_transitions(i, max_transitions)
        for i in range(len(train_set))
    ]
    resolution_hist = defaultdict(int)
    for transitions in resolved_transitions_by_local:
        resolution_hist[int(transitions)] += 1
    hist_summary = ", ".join(
        f"{t}:{resolution_hist[t]}" for t in sorted(resolution_hist.keys())
    )
    print(
        "Dense transition max-adaptive buckets (transitions:count): "
        f"{hist_summary}"
    )

    batch_size = int(cfg.loader.batch_size)
    drop_last = bool(cfg.loader.get("drop_last", True))
    train_batch_sampler = FixedCoverageMaxTransitionsBatchSampler(
        num_samples=len(train_set),
        batch_size=batch_size,
        resolved_transitions_by_local=resolved_transitions_by_local,
        drop_last=drop_last,
        seed=int(cfg.seed),
    )

    train_loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    train_loader_kwargs.pop("batch_size", None)
    train_loader_kwargs.pop("shuffle", None)
    train_loader_kwargs.pop("drop_last", None)

    train = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        generator=rnd_gen,
        **train_loader_kwargs,
    )

    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    predictor = ARPredictor(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_embedder_kwargs = {
        "use_positional_encoding": bool(
            cfg.wm.get("action_use_positional_encoding", True)
        )
    }
    action_prefix_cfg = cfg.wm.get("action_prefix", None)
    if action_prefix_cfg is not None:
        action_embedder_kwargs.update(
            OmegaConf.to_container(action_prefix_cfg, resolve=True)
        )

    action_encoder = Embedder(
        input_dim=cfg.wm.action_dim,
        emb_dim=embed_dim,
        **action_embedder_kwargs,
    )
    action_encoder_impl = getattr(action_encoder, "impl", action_encoder)
    setattr(action_encoder_impl, "action_base_dim", int(cfg.wm.action_base_dim))
    setattr(action_encoder_impl, "action_frame_skip", int(cfg.wm.action_frame_skip))

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        default_root_dir=str(run_dir),
        enable_checkpointing=True,
        use_distributed_sampler=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
