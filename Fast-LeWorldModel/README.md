<h1 align="center">Fast LeWorldModel</h1>

<p align="center">
  <a href="https://yuntian-gao.github.io/"><strong>Yuntian Gao</strong></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://xuxy09.github.io/"><strong>Xiangyu Xu</strong></a><sup>*</sup>
  <br>
  <sub><sup>*</sup> Corresponding Author</sub>
</p>



<p align="center">
  <a href="https://fast-lewm.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://github.com/Yuntian-Gao/Fast-LeWorldModel">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code">
  </a>
  <a href="https://arxiv.org/abs/2606.26217">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
  </a>
</p>

Official implementation of **Fast-LeWorldModel**.
## Installation

```bash
git clone https://github.com/Yuntian-Gao/Fast-LeWorldModel.git
cd Fast-LeWorldModel

conda create -n fast-lewm python=3.10 -y
conda activate fast-lewm
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

## Data

Use the same datasets and layout as
[LeWM](https://github.com/lucas-maes/le-wm#data). Place the extracted `.h5`
files under one directory and set:

For example:

```text
stablewm_data/
├── pusht_expert_train.h5
├── tworoom_expert_train.h5
├── reacher_expert_train.h5
└── cube_expert_train.h5
```
set the
`STABLEWM_HOME` environment variable:
```bash
export STABLEWM_HOME=/absolute/path/to/stablewm_data
```

## Pretrained Checkpoints

Pretrained Fast-LeWorldModel checkpoints are available on Hugging Face:

- [naiverer/fast-leworldmodel](https://huggingface.co/naiverer/fast-leworldmodel)


## Training

Set your Weights & Biases `entity` and `project` in
`config/train/Fast-lewm.yaml`:

```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

Launch training with one of the configs under `config/train/data/`:

```bash
python train.py data=pusht
```

Hydra writes checkpoints to `outputs/<date>/<time>/` by default, including:

```text
outputs/<date>/<time>/Fast-lewm_weights.ckpt
outputs/<date>/<time>/Fast-lewm_epoch_xx_object.ckpt
```


## Planning and Evaluation

```bash
python eval.py --config-name=pusht \
  eval.dataset_path=/absolute/path/to/pusht_expert_train.h5 \
  eval.ckpt_path=/absolute/path/to/Fast-lewm_pusht_object.ckpt
```

Available evaluation configs are `pusht`, `tworoom`, `reacher`, and `cube`.

## Acknowledgements

This codebase is built on the official
[LeWorldModel](https://github.com/lucas-maes/le-wm) implementation. We thank
the authors of LeWM for releasing their codebase.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{gao2026fast,
  title={Fast LeWorldModel},
  author={Gao, Yuntian and Xu, Xiangyu},
  journal={arXiv preprint arXiv:2606.26217},
  year={2026}
}
```
