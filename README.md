# Backdoor Attacks in Federated Learning

Abstract
--

Federated Learning is a decentralized machine learning approach that allows multiple clients (devices, servers, or organizations) to collaboratively train a global model without sharing their private data with a central server.

Because of the distributed nature of Federated Learning, the central server cannot verify the integrity of training data. It exposed the system to poisoning attacks by malicious clients. Some clients could inject a trigger into images of the target class but does not swap the label (**Clean-Label Attacks**) or insert a trigger into an image and flips the label of that image to the target class (**Dirty-Label Attacks**).

Objectives
--

- Investigate how IID and non-IID data distributions influence the success of backdoor attacks.

- Implement several attack strategies (label-poisoning, clean-label, colluding attackers).

- Quantify the trade-off between Attack Success Rate (ASR) and model utility under different aggregation strategies (FedAvg, norm bounding).

Approach
--
1. Simulate FL training on standard datasets (MNIST, CIFAR-10, GTSRB).

2. Introduce controlled levels of data heterogeneity across clients.

3. Inject various backdoor attack types and analyze performance/stealthiness.

4. Provide insights into how AI security is compromised by such attacks.

