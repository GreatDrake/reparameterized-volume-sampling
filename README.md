# reparameterized-volume-sampling

Code release for paper [Differentiable Rendering with Reparameterized Volume Sampling](https://arxiv.org/abs/2302.10970).

<img align="middle" width="70%" src="figs/spline_inversion.png">

## 1D Experiments

Notebook `1d_experiments.ipynb` contains RVS sampling algorithm implementation and single ray experiments from the paper.

## Differentiable Hierarchical Sampling

Our NeRF implementation is based on [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) repository and follows the same structure. 

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

## Monte Carlo Estimates

Under construction 🚧 🔨

## Citation

```
@article{morozov2023differentiable,
  title={Differentiable Rendering with Reparameterized Volume Sampling},
  author={Morozov, Nikita and Rakitin, Denis and Desheulin, Oleg and Vetrov, Dmitry and Struminsky, Kirill},
  journal={arXiv preprint arXiv:2302.10970},
  year={2023}
}
```
