# CIEGCL
This is the PyTorch implementation for CIEGCL proposed in the paper CIEGCL: Counterfactual Intervention Enhancing Graph Contrastive Learning in Implicit Feedback

### 1. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
torch==1.13.1
numpy==1.21.5
pandas=1.3.5
tqdm==4.64.1
```

### 2. Some configurable arguments

* `--topks` Top k for testing recommendation performance.
* `--cl_weight` specifies $\lambda_S$, the regularization weight for CL loss.
* `--weight_decay` is $\lambda$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--static_prob` is the edge dropout rate.
* `--batch_size` is the train batch size.
* `--embedding_dim` is the embedding dimension for embedding based models.
* `--n_layers` is the layer number for GNN models.

### 3. Different from the default parameter settings

* Ciao
```
--cl_weight 0.01
```

* Movielens-1M
```
--cl_weight 0.01 --temp 0.2
```

* Coat

```
--batch_size 64 --temp 0.5 --weight_decay 0.01
```
