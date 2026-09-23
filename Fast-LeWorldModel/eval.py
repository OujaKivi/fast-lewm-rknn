import os

os.environ["MUJOCO_GL"] = "egl"

import ast
import inspect
from collections import defaultdict
from functools import wraps
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from transformers import ViTConfig, ViTModel
import stable_worldmodel as swm


def _repair_released_vit(model):
    """Rebuild the released pickled ViT for newer Transformers versions."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or hasattr(encoder.encoder.layer[0].attention.attention, "dropout"):
        return model
    rebuilt = ViTModel(
        ViTConfig(
            hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
            intermediate_size=768, patch_size=14, image_size=224,
            num_channels=3,
        ),
        add_pooling_layer=False,
    )
    rebuilt.load_state_dict(encoder.state_dict(), strict=True)
    model.encoder = rebuilt
    return model


def _install_fast_buffered_action_path(policy_obj):
    """Skip image preprocessing on MPC steps that only consume buffered actions."""
    if not all(
        hasattr(policy_obj, name)
        for name in ["get_action", "_action_buffer", "process", "env"]
    ):
        return None

    original_get_action = policy_obj.get_action

    @wraps(original_get_action)
    def fast_get_action(info_dict, **kwargs):
        assert hasattr(policy_obj, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        if getattr(policy_obj, "_action_buffer", None) is None:
            return original_get_action(info_dict, **kwargs)

        # When MPC already has actions buffered, no model input is needed.
        # Avoid expensive pixels/goal transforms and just execute the next action.
        if len(policy_obj._action_buffer) == 0:
            return original_get_action(info_dict, **kwargs)

        action = policy_obj._action_buffer.popleft()
        action = action.reshape(*policy_obj.env.action_space.shape)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action)

        if "action" in policy_obj.process:
            action = policy_obj.process["action"].inverse_transform(action)

        return action

    policy_obj.get_action = fast_get_action
    print("[eval] enabled fast buffered action path: skip _prepare_info when CEM is not needed")
    return original_get_action


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name, dataset_cache_dir=None, dataset_path=None):
    kwargs = {
        "keys_to_cache": cfg.dataset.keys_to_cache,
    }
    if dataset_path is not None:
        kwargs["path"] = Path(dataset_path).expanduser()
    else:
        kwargs["name"] = dataset_name
        kwargs["cache_dir"] = (
            dataset_cache_dir
            or Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        )
    dataset = swm.data.HDF5Dataset(**kwargs)
    return dataset


def parse_dataset_reference(cfg):
    """Return dataset_name + cache_dir, supporting explicit dataset file path."""
    dataset_ref = cfg.eval.get("dataset_path")
    if dataset_ref:
        dataset_ref = Path(dataset_ref).expanduser()
        dataset_name = dataset_ref.stem if dataset_ref.suffix in [".hdf5", ".h5"] else dataset_ref.name
        return dataset_name, dataset_ref.parent

    dataset_name = cfg.eval.dataset_name
    dataset_cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return dataset_name, dataset_cache_dir


def parse_policy_reference(cfg):
    """
    Return policy identifier + cache_dir, supporting explicit checkpoint path.
 
    """
    policy_ref = cfg.get("policy", "random")
    cache_dir = cfg.get("cache_dir")
    if policy_ref == "random":
        return "random", cache_dir

    ckpt_path = cfg.eval.get("ckpt_path")
    if ckpt_path:
        ckpt_path = Path(ckpt_path).expanduser()
        run_name = ckpt_path.name
        if run_name.endswith("_object.ckpt"):
            run_name = run_name[: -len("_object.ckpt")]
        elif run_name.endswith(".ckpt"):
            run_name = run_name[: -len(".ckpt")]
        return run_name, str(ckpt_path.parent)

    return policy_ref, cache_dir


def _find_task_column(dataset):
    candidates = ["task", "task_id", "task_name", "env_task", "env_id"]
    for col in candidates:
        if col in dataset.column_names:
            return col
    return None


def _to_task_key(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _build_task_labels(dataset, row_indices, default_task="all"):
    task_col = _find_task_column(dataset)
    row_data = dataset.get_row_data(row_indices)
    labels = []
    if task_col is None or task_col not in row_data:
        labels = [str(default_task) for _ in range(len(row_indices))]
        return labels, task_col

    for v in np.asarray(row_data[task_col]):
        labels.append(str(_to_task_key(v)))
    return labels, task_col


def _build_loss_metrics(cost_trace, task_labels, task_col, source):
    if len(cost_trace) == 0:
        return None

    cost_mat = np.stack(cost_trace, axis=0).astype(np.float64)
    if cost_mat.ndim == 1:
        cost_mat = cost_mat[None, :]

    per_env_cost = cost_mat.mean(axis=0)
    by_task_values = defaultdict(list)
    for task_name, loss in zip(task_labels, per_env_cost.tolist()):
        by_task_values[str(task_name)].append(float(loss))

    by_task = {}
    for task_name, values in by_task_values.items():
        arr = np.asarray(values, dtype=np.float64)
        by_task[task_name] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    all_arr = np.asarray(per_env_cost, dtype=np.float64)
    return {
        "source": source,
        "task_column": task_col,
        "num_replans": int(cost_mat.shape[0]),
        "count": int(all_arr.size),
        "mean": float(all_arr.mean()),
        "std": float(all_arr.std()),
        "by_task": by_task,
    }


def _resolve_eval_plan_config(cfg: DictConfig):
    plan_cfg = cfg.plan_config
    horizon = int(plan_cfg.get("horizon", 1))
    action_num_blocks = int(plan_cfg.get("action_num_blocks", plan_cfg.get("horizon", 1)))
    action_block_size = int(plan_cfg.get("action_block_size", plan_cfg.get("action_block", 1)))
    receding_horizon = int(plan_cfg.get("receding_horizon", 1))

    return horizon, action_num_blocks, action_block_size, receding_horizon


def _parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    elif torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[0] in "[(":
            value = ast.literal_eval(text)
        else:
            value = text.split(",")
    elif isinstance(value, (int, float)):
        value = [value]

    values = [int(v) for v in value]
    if len(values) == 0:
        return None
    return values


def _resolve_rollout_consistency_config(cfg: DictConfig, action_num_blocks: int):
    plan_cfg = cfg.plan_config
    consistency_weight = float(
        plan_cfg.get(
            "consistency_loss_weight",
            0.0
        )
    )
    blocks_per_step = _parse_int_list(plan_cfg.get("action_num_blocks_per_step", None))

    return consistency_weight, blocks_per_step


def _configure_rollout_consistency(model, consistency_weight, blocks_per_step):
    targets = [model]
    if isinstance(model, torch.nn.Module):
        targets.extend(
            m for m in model.modules() if m is not model and hasattr(m, "get_cost")
        )

    for target in targets:
        setattr(target, "consistency_loss_weight", float(consistency_weight))
        setattr(target, "action_num_blocks_per_step", blocks_per_step)

    if consistency_weight != 0.0:
        print(
            "[eval] rollout consistency cost enabled: "
            f"weight={consistency_weight}, action_num_blocks_per_step={blocks_per_step}"
        )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    horizon, action_num_blocks, action_block_size, receding_horizon = _resolve_eval_plan_config(cfg)
    action_block = int(action_num_blocks * action_block_size)
    assert (
        horizon * action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    # Current stable-worldmodel no longer consumes these legacy World keys;
    # forwarding them would incorrectly pass them into PushT.__init__().
    world_cfg.pop("history_size", None)
    world_cfg.pop("frame_skip", None)
    world = swm.World(**world_cfg, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset_name, dataset_cache_dir = parse_dataset_reference(cfg)
    dataset = get_dataset(
        cfg,
        dataset_name,
        dataset_cache_dir=dataset_cache_dir,
        dataset_path=cfg.eval.get("dataset_path"),
    )
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy_name, policy_cache_dir = parse_policy_reference(cfg)
    consistency_weight, blocks_per_step = _resolve_rollout_consistency_config(
        cfg, action_num_blocks
    )

    if policy_name != "random":
        model = swm.policy.AutoCostModel(policy_name, cache_dir=policy_cache_dir)
        model = _repair_released_vit(model)
        model = model.to(cfg.solver.device)
        model = model.eval()
        model.requires_grad_(False)
        _configure_rollout_consistency(model, consistency_weight, blocks_per_step)

       
        for m in model.modules():
            if isinstance(m, torch.nn.GRU):
                m.train()
        model.interpolate_pos_encoding = True
        plan_cfg = OmegaConf.to_container(cfg.plan_config, resolve=True)
        plan_cfg["horizon"] = int(horizon)
       
        plan_cfg["action_block"] = int(action_block)
        plan_cfg["receding_horizon"] = int(receding_horizon)
     
        valid_plan_keys = set(inspect.signature(swm.PlanConfig).parameters.keys())
        plan_cfg = {k: v for k, v in plan_cfg.items() if k in valid_plan_keys}
        config = swm.PlanConfig(**plan_cfg)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy_obj = swm.policy.RandomPolicy()

    results_path = (
        Path(policy_cache_dir or swm.data.utils.get_cache_dir(), policy_name).parent
        if policy_name != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

   
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy_obj)
    # Keep the installed stable-worldmodel policy path intact for correctness;
    # its action buffer is per-environment in current releases.
    fast_get_action_backup = None

    latent_loss_metrics = None
    cem_trace = {"optimized_costs": []}
    solver_solve_backup = None
    task_labels, task_col = _build_task_labels(
        dataset=dataset,
        row_indices=random_episode_indices,
        default_task=cfg.world.get("task", "all"),
    )
    if policy_name != "random" and hasattr(model, "get_cost") and hasattr(policy_obj, "solver"):
        solver = policy_obj.solver
        solver_solve_backup = solver.solve

        def _wrapped_solve(info_dict, init_action=None):
            outputs = solver_solve_backup(info_dict, init_action=init_action)

            try:
                # CEMSolver already computes optimized latent costs in outputs["costs"]
                # (final-iteration top-k mean per env), avoid re-calling get_cost here.
                if "costs" in outputs:
                    cem_trace["optimized_costs"].append(
                        np.asarray(outputs["costs"], dtype=np.float64)
                    )
            except Exception as exc:
                print(f"[warn] failed to record CEM latent cost: {exc}")

            return outputs

        solver.solve = _wrapped_solve

    try:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
    finally:
        if solver_solve_backup is not None:
            policy_obj.solver.solve = solver_solve_backup
        if fast_get_action_backup is not None:
            policy_obj.get_action = fast_get_action_backup

    cost_shapes = {np.asarray(cost).shape for cost in cem_trace["optimized_costs"]}
    latent_loss_metrics = (
        _build_loss_metrics(
            cem_trace["optimized_costs"],
            task_labels=task_labels,
            task_col=task_col,
            source="cem_optimized_latent_cost",
        )
        if len(cost_shapes) <= 1 else None
    )
    if latent_loss_metrics is not None:
        print("latent_loss_metrics", latent_loss_metrics)

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        if latent_loss_metrics is not None:
            f.write(f"latent_loss_metrics: {latent_loss_metrics}\n")


if __name__ == "__main__":
    run()
