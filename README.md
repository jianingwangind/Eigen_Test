# Eigen_Test
Eigen running Tests
# Active Learning
* **Query-synthesizing** approaches: which use generative models to generate informative samples.#
* **Query-acquiring** approaches: use different sampling strategies to select the most informative samples.
  - Uncertainty-based approaches: examples which are the most uncertain (output probabilities near 0.5) by the model will be annotated by an oracle.
  - Representation-based approaches: which rely on selecting few examples by increasing diversity in a given batch.
  - Their combinations.
# Latest Papers
* [Discriminative Active learning](https://arxiv.org/pdf/1907.06347v1.pdf):
  The active learning objective is posed as a binary classification problem and attempts to make the labeled set indistinguishable from the unlabeled pool. For the binary classification, a multi-layered perceptron with 3 hidden layers is chosen.
