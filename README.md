# Tensor Parametric Hamiltonian Operator Inference

This repository contains the source code for the numerical experiments in the paper

[**Tensor parametric Hamiltonian operator inference**](https://arxiv.org/abs/2502.10888)

by\
[Arjun Vijaywargiya](https://scholar.google.com/citations?user=_fcSwDYAAAAJ) (Notre Dame and Sandia National Laboratories),\
[Shane A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ) (Sandia National Laboratories), and\
[Anthony Gruber](https://scholar.google.com/citations?user=CJVuqfoAAAAJ) (Sandia National Laboratories).

<details><summary>BibTex</summary><pre>
@misc{vijaywargiya2025tensorpopinf,
    title = {Tensor parametric Hamiltonian operator inference},
    author = {Arjun Vijaywargiya and Shane A. McQuarrie and Anthony Gruber},
    year = {2025},
    eprint = {2502.10888},
    archivePrefix = {arXiv},
}
</pre></details>

## Contents

Methodology

- [tensor_inference.py](./tensor_inference.py): tensor-based inference algorithms listed in the paper.
- [models.py](./models.py): Tensor-based nonintrusive inference of parametric reduced-order models, extending the [`opinf`](https://willcox-research-group.github.io/rom-operator-inference-Python3/source/index.html) package.

Example 1: Heat equation

- [heatEq.py](./heatEq.py): full-order models for the heat equation with piecewise constant diffusion.
- [Heat1D.ipynb](./Heat1D.ipynb): numerical experiments for the heat equation in one spatial dimension.
- [Heat2D.ipynb](./Heat2D.ipynb): numerical experiments for the heat equation in two spatial dimensions.

Example 2: Wave equation

- [waveEq.py](./waveEq.py): full-order models for the wave equation with piecewise constant wave speed.
- [Wave1D.ipynb](./Wave1D.ipynb): numerical experiments for the wave equation in one spatial dimension.
- [Wave2D.ipynb](./Wave2D.ipynb): numerical experiments for the wave equation in two spatial dimensions.
- [WaveBlowup.ipynb](./WaveBlowup.ipynb):

Other

- [utils.py](./utils.py): utilities for computing errors and plotting figures.
- [requirements.txt](./requirements.txt): exact python environment used for the paper.

## Installation

This repository uses the standard Python scientific stack (NumPy, SciPy, Scikit-Learn, etc.), the [`ngsolve`](https://ngsolve.org/) finite element library, and the [`opinf`](https://willcox-research-group.github.io/rom-operator-inference-Python3) package.
We recommend installing the required packages in a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```shell
$ conda deactivate
$ conda create -n tensoropinf python=3.13
$ conda activate tensoropinf
(tensoropinf) $ pip install -r requirements.txt
```

## Related Work

Other branches in this repository correspond to related papers.

- `main`: [**Canonical and noncanonical Hamiltonian operator inference**](https://www.sciencedirect.com/science/article/pii/S0045782523004589) by Gruber and Tezaur, CMAME 2023
- `var-consistent`: [**Variationally consistent Hamiltonian model reduction**](https://epubs.siam.org/doi/full/10.1137/24M1652490) by Gruber and Tezaur, SIADS 2024.
