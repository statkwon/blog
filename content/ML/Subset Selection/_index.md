---
title: "Subset Selection"
date: 2021-02-22
weight: 4
---

**Best-Subset Selection**

{{<figure src="/esl_fig_3.5.png" width="500" height="200">}}

If we have $p$ variables, we can consider $2^p$ models. For each number of variables, we will select a model with smallest residual sum of squares. The red dots in the figure above are the selected ones. Among these $p+1$ models, we have to choose the best model. Here, the best is usually measured by cross-validation error or AIC criterion. It is reasonable to pick a smaller model when the scores are almost similar. This is a quite accurate method as we search through all possible subsets of variables, but is also a time-consuming procedure.

---

**Forward-Stepwise Selection**

To deal with the calculation cost of best-subset selection, Forward- or Backward-Stepwise Selection can be suggested. Forward-Stepwise Selection differs from best-subset selection in that it starts from the model with only the intercept term and add one variable at a time which reduces the residual sum of squares most. Among $p+1$ models from each step, the best model is selected by the same way used in best-subset selection. This process results in fitting $1+p+(p-1)+\cdots+1=1+\dfrac{p(p+1)}{2}$ models in total, so it is certainly a cost-efficient method.

This greedy algorithm can be clever by using $QR$-Decomposition.

---

**Backward-Stepwise Selection**

Backward-Stpewise Selection is almost same as forward-stepwise method, except that it starts with the full model. It removes one variable at a time which reduces the residual sum of squares most.

---

**Forward-Stagewise Regression**

---

**Reference**

1. Elements of Statistical Learning