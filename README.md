<h1 align="center">
  <b>Unitrans</b><br>
</h1>

**Paper: UniTrans: A Unified Vertical Federated Knowledge Transfer Framework for Vulnerable Patient Groups**

### 1. Overview

![](assets/overview.png)

### 2. Install

```
conda install --file requirements.txt
```

### 3. Run

```
python --dataset mimic --type intra --frl FedSVD --lkt AE --task_model xgboost
```