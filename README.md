# Invariant EEG Representation Learning via Adversarial Inference

This is the invariant EEG representation learning convolutional neural network (CNN) implementation, which uses an adversarial training scheme to monitor and/or censor nuisance-specific leakage in learned representations. The aim is to perform adversarial censoring during the CNN training procedure such that the network can learn nuisance-invariant representations within the discriminative CNN setting. Implementation is in Python using Keras with the Tensorflow backend.

# Usage

An example execution is as follows:

```python

from AdversarialCNN import AdversarialCNN

net = AdversarialCNN(chans = ..., samples = ..., n_output = ..., n_nuisance = ..., architecture = ..., adversarial = ..., lam = ...)

net.train(training_set, validation_set, log = ..., epochs = ..., batch_size = ...)

```

Boolean parameter `adversarial = True` trains the network via adversarial censoring. If `False`, then an adjacent adversary network is simply trained to monitor nuisance-specific leakage in the representations. Parameter `lam` indicates the adversarial regularization weight embedded in the loss function. Parameter `architecture` defines the regular CNN blocks excluding the final dense layer (i.e., linear classification layer), which can be modified arbitrarily. Default implementations include `architecture = 'EEGNet'` from (Lawhern et al. 2018), as well as `architecture = 'DeepConvNet'` and `architecture = 'ShallowConvNet'` from (Schirrmeister et al. 2017), which are well-known CNN architectures used for EEG classification.

# Paper Citation Note
Codes will be available soon after manuscript revisions.