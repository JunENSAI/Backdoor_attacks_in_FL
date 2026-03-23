# Federated Clean-Label Backdoor (FCLB): Implementation, Evaluation, and Countermeasures

## Abstract

Federated Learning (FL) is a decentralized paradigm enabling multiple clients to collaboratively train a global model without sharing raw private data. While this architecture enhances data privacy, it introduces a critical security vulnerability: the central server cannot inspect local datasets and must trust client updates. This limitation opens the door to training-time attacks, notably backdoor attacks.

This repository presents an empirical evaluation of the Federated Clean-Label Backdoor (FCLB) attack. This stealthy approach injects a hidden trigger into the global model while preserving correct labels, allowing it to evade both manual inspection and automated data sanitization. We analyze the trade-off between Attack Success Rate (ASR) and primary task accuracy across different data distributions (IID and Non-IID), and evaluate robustness against Byzantine-resilient aggregation defenses.

## Key Contributions

- **Empirical Extension to Grayscale Domains**  
  Extension of the FCLB evaluation to the MNIST dataset, demonstrating that clean-label backdoor mechanisms remain effective even in constrained feature spaces.

- **Verification of the Lazy Attack State**  
  Reconstruction of the FCLB pipeline and empirical validation of the periodic unlearning phase required to sustain backdoor persistence.

- **Controlled IID Baseline Evaluation**  
  Evaluation under IID conditions to isolate intrinsic FCLB behavior and approximate upper bounds on attack success rate and convergence.

- **Aggressive Stress-Testing**  
  Use of a 30% attacker ratio on CIFAR-10 to assess the impact of a heavily compromised federated network.

## Methodology and Threat Model
The FCLB attack assumes a realistic medium-threat setting in which the adversary controls a subset of clients without access to benign data or aggregation mechanisms.

The attack consists of three main stages:

1. **Trigger Generation via Surrogate Model**  
   A pre-trained surrogate model combined with Public Out-of-Distribution (POOD) data is used to craft an $l_{\infty}$-bounded perturbation targeting a specific class.

2. **Deviation Reduction**  
   During local training, malicious clients incorporate an $l_{2}$ penalty to constrain updates close to the global model parameters, reducing detectability.

3. **Targeted Unlearning Schedule**  
   To maintain backdoor effectiveness over time, clients periodically perform gradient ascent on clean target samples (e.g., every 3 rounds for MNIST).

## Experimental Setup
- **Datasets**: MNIST (target class: 7), CIFAR-10 (target class: 8)  
- **Data Partitioning**:
  - IID (Independent and Identically Distributed)
  - Non-IID using symmetric Dirichlet distribution  
- **Baseline**: Constrain-and-Scale dirty-label backdoor attack  

## Defenses Evaluated
- **Krum Aggregation**  
  A clustering-based method evaluating the proximity of local updates. Experimental results show limited effectiveness against FCLB.

- **Norm Clipping**  
  Imposes a maximum $l_{2}$ norm constraint on updates. With a threshold $M = 5.0$, this method reduces ASR to near zero while preserving main task performance.

## References

[1] Xie, Y., & Zhu, T. (2024). *Clean-label Backdoor Attack in Federated Learning*. In *2024 5th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)* (pp. 139–148). IEEE.  
Provides the foundational framework for the FCLB attack.

[2] Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020). *How to Backdoor Federated Learning*. In *International Conference on Artificial Intelligence and Statistics* (pp. 2938–2948). PMLR.  
Introduces the Constrain-and-Scale dirty-label attack used as a comparative baseline.

[3] Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). *Federated Machine Learning: Concept and Applications*. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 10(2), 1–19.  
Provides foundational context on federated learning systems and privacy-preserving training.

[4] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*. *Advances in Neural Information Processing Systems*, 30.  
Introduces the Krum aggregation method evaluated among the defenses.

[5] Sun, Z., Kairouz, P., Suresh, A. T., & McMahan, H. B. (2019). *Can You Really Backdoor Federated Learning?* arXiv:1911.07963.  
Describes the Norm Clipping defense used to constrain malicious updates.