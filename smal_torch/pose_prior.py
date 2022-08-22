import pickle as pkl
import numpy as np
from chumpy import Ch
import torch

class Poseprior(object):
    def __init__(self, prior_path, device):
        with open(prior_path, "rb") as f:
            res = pkl.load(f, encoding='latin1')

        self.precs = torch.from_numpy(res['pic'].r.copy()).float().to(device)
        self.mean = torch.from_numpy(res['mean_pose']).float().to(device)

        # Ignore the first 3 global rotation.
        prefix = 3
        pose_len = 108

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False

        self.use_ind_tch = torch.from_numpy(self.use_ind).float().to(device)

    def __call__(self, x):
        mean_sub = x.reshape(-1, 36 * 3)[:, self.use_ind] - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims=([1], [0]))
        return res ** 2


def test_prior():
    device = torch.device('cuda:0')
    pose = Poseprior(
        prior_path='../smpl_models/walking_toy_symmetric_smal_0000_new_skeleton_pose_prior_new_36parts.pkl',
        device=device)
    data = torch.randn([5,108]).to(device)
    data.requires_grad = True
    optimizer = torch.optim.Adam([data], lr=0.01, betas=(0.9, 0.999))
    for i in range(300):
        loss = pose(data).mean()

        optimizer.zero_grad()
        # compute new grad parameters through time!
        loss.backward()
        optimizer.step()
        print(f'epoch {i} loss {loss.item()}')


if __name__ == '__main__':
    test_prior()