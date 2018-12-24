import numpy as np
import torch


def noise_generator(train_labels, noise_rate=0., noise_type='symmetry'):
    print('Generating noisy label, noise_rate %f, noise_type %s' % (noise_rate, noise_type))
    label = train_labels.clone()
    num = label.__len__()
    d_label = max(label) + 1
    p_symmetry = [1/max(label) for _ in range(d_label)]

    if noise_type == 'symmetry':
        for c in range(d_label):
            prob = p_symmetry
            prob[c] = 0
            idx = np.where(train_labels == c)
            sample_idx = np.random.choice(idx[0], int(len(idx[0]) * noise_rate), replace=False)
            for i in range(len(sample_idx)):
                label[sample_idx[i]] = torch.Tensor(np.random.choice(d_label, 1, prob))

    elif noise_type == 'pair':
        for c in range(d_label):
            idx = np.where(train_labels == c)
            sample_idx = np.random.choice(idx[0], int(len(idx[0]) * noise_rate), replace=False)
            for i in range(len(sample_idx)):
                label[sample_idx[i]] = (c + 1) % d_label.item()
    else:
        AssertionError("noise_type is error!")

    rate = np.where(label != train_labels)[0].shape[0] / num
    print('Exactly noisy rate is %f' % rate)
    return label