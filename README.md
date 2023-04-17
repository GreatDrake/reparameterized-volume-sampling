# reparameterized-volume-sampling

Code release for paper [Differentiable Rendering with Reparameterized Volume Sampling](https://arxiv.org/abs/2302.10970).

Repository is still under construction 🚧 🔨

## 1D Experiments

Notebook `1d_experiments.ipynb` contains RVS sampling algorithm implementation and single ray experiments from the paper.

## Differentiable Hierarchical Sampling

Our NeRF implementation is based on [nerf-pytorch](https://github.com/google/mipnerf) repository and follows the same structure. 

Training NeRF with RVS algorithm for differentiable hierarchical sampling: 

```
cd nerf
python run_nerf.py --config configs/{SCENE}_rvs.txt
```

replace `{SCENE}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

Testing:

```
python run_nerf.py --config configs/{SCENE}_rvs.txt --render_only
```
