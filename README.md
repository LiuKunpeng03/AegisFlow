# AegisFlow

<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/LiuKunpeng03/AegisFlow?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/LiuKunpeng03/AegisFlow?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.10-purple.svg" alt="Python">

</div>

### AegisFlow: Training-free Non-myopic Path-safe Guided Flow Matching

*Kunpeng Liu, Boshi Zhang, Keyou You†*

† Corresponding author.

In this paper, we propose AegisFlow, a training-free, non-myopic path-safe guidance framework that recasts guided generation as a safety-critical optimal control problem with global scope. By integrating a time-varying barrier function directly into the global cost functional, AegisFlow naturally enforces constraints along the entire generative path. To compute the global optimal control, we derive a tractable, mathematically grounded approximation of the Hamilton-Jacobi-Bellman (HJB) equation.

<p align="center">
  <img src="pictures/concept.png" alt="Triangle plot" width="100%"/>
</p>

This branch (`planning`) reproduces the planning experiments from the paper.

---

### Installation

```bash
conda env create -f environment.yml
conda activate aegisflow_planning
pip install -e .
pip install qpth cvxpy cvxopt
pip install torchdyn torchdiffeq torchcfm
pip install git+https://github.com/atong01/conditional-flow-matching.git
```

---

### Training

Run the following line to train the generative prior.

```bash
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1 --method cfm
```

---

### Evaluation

```bash
bash scripts/test.sh
```
The results can be found in `./logs/maze2d-large-v1`

---

### Acknowledgement

- The maze navigation implementation is based on [SafeFlowmatcher](https://github.com/takahashi-seiryu/SafeFlowMatcher).