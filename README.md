# Invariant EEG Representation Learning via Adversarial Inference

This is the invariant EEG representation learning convolutional neural network (CNN) implementation, which uses an adversarial training scheme to monitor and/or censor nuisance-specific leakage in learned representations. The aim is to perform adversarial censoring during the CNN training procedure such that the network can learn nuisance-invariant representations within the discriminative CNN setting. Implementation is in Python using Keras with the Tensorflow backend.

# Usage

An example execution is as follows:

```python

from AdversarialCNN import AdversarialCNN

net = AdversarialCNN(chans = ..., samples = ..., n_output = ..., n_nuisance = ..., architecture = ..., adversarial = ..., lam = ...)

net.train(train_set, validation_set, log = ..., epochs = ..., batch_size = ...)

```

Boolean parameter `adversarial = True` trains the network via adversarial censoring. If `False`, then an adjacent adversary network is simply trained to monitor nuisance-specific leakage in the representations. Parameter `lam` indicates the adversarial regularization weight embedded in the loss function. Parameter `architecture` defines the regular CNN blocks excluding the final dense layer (i.e., linear classification layer), which can be modified arbitrarily. Default implementations include `architecture = 'EEGNet'` from (Lawhern et al. 2018), as well as `architecture = 'DeepConvNet'` and `architecture = 'ShallowConvNet'` from (Schirrmeister et al. 2017), which are well-known CNN architectures used for EEG classification.

To use the `train` function, both `train_set` and `validation_set` should be three-element tuples (i.e., `x_train, y_train, s_train = train_set`). Here, the first element `x_train` is the EEG data of size `(num_observations, num_channels, num_timesamples, 1)`, `y_train` are the one-hot encoded class labels (e.g., for binary labels will have a size `(num_observations, 2)`), and `s_train` are the one-hot encoded nuisance labels (e.g., for 10-class nuisance labels will have size `(num_observations, 10)`). Variable `log` indicates the directory string to save the log files during training.

# Paper Citation
If you use this code in your research and find it helpful, please cite the following paper:
> Ozan Ozdenizci, Ye Wang, Toshiaki Koike-Akino, Deniz Erdogmus. "Learning Invariant Representations from EEG via Adversarial Inference". IEEE Access, 2020.

# Acknowledgments
Ozan Ozdenizci and Deniz Erdogmus are partially supported by NSF (IIS-1149570, CNS-1544895, IIS-1715858), DHHS (90RE5017-02-01), and NIH (R01DC009834).
