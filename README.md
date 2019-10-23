# Active Learning

## Categories by Sampling Strategies
* **Query-synthesizing** approaches: which use generative models to generate informative samples.#

* **Query-acquiring** approaches: use different sampling strategies to select the most informative samples.

  - Uncertainty-based approaches: examples which are the most uncertain (output probabilities near 0.5) by the model will be annotated by an oracle.
  - Representation-based approaches: which rely on selecting few examples by increasing diversity in a given batch.
  - Their combinations.
## Latest Papers
* [Discriminative Active learning](https://arxiv.org/pdf/1907.06347v1.pdf):
  The active learning objective is posed as a binary classification problem and attempts to make the labeled set indistinguishable from the unlabeled pool. For the binary classification, a multi-layered perceptron with 3 hidden layers is chosen.

* [Self-Paced Active Learning](https://aaai.org/ojs/index.php/AAAI/article/view/4445): 
  Besides the informativeness and the representativeness of an instance on which the most papers only focus, in this paper, whether the selected example can be fully utilized by the current model is investigated. These two aspects are incorporated into a unified framework of self-paced active learning.
  
* [Variational Adversarial Active Learning](https://arxiv.org/pdf/1904.00370v2.pdf) (state-of-art):
  Using query-acquiring approach, the sampling mechanism is implicitly learned by a VAE and a discriminator in adversarial manner. The uncertainty for the samples deemed to be from the unlabeled pool is modeled.
  
* [Active Learning with Partial Feedback](https://arxiv.org/pdf/1802.07427v4.pdf) (large-scale multiclass classification):
  At each training step, the learner selects an example, asking if it belongs to a chosen class to exploit class hierarchies to drill down to the exact label. The annotator then responses with binary feedback.
  
# Tips
  In most papers, different back-bone networks are used. For classification, ResNet-18 or VGG16 are widely used. In my opinion, the researchers focus on how to select the most informative and representative examples effectively and efficiently.
