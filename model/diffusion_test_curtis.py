import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# Data parameters
MEANS = [torch.tensor([-0.5, 0.5]), torch.tensor([0.5, -0.5])]
VARIANCES = [torch.eye(2)*0.1]*2
WEIGHTS = [0.5, 0.5]
PAIR_N = 2000

LR = 1e-3
B1 = 0.99
N_HIDDEN = 400
N_ITER = int(1.5e5)
K_BINS = 50

torch.manual_seed(42)

def gaussian_mixture(means, covariances, weights, n_samples): 
    samples = []
    for i in range(len(means)):
        m = means[i]
        c = covariances[i]
        dist = D.MultivariateNormal(m, covariance_matrix=c)
        samples.append(dist.sample((n_samples,)))

    indicies = torch.randint(0, len(means), (n_samples,))
    output = torch.zeros((n_samples, len(means[0])))
    for i in range(n_samples):
        output[i] = samples[indicies[i]][i]
        
    return output, indicies

def make_data(means, variances, weights, pair_n):
    pair_data, pair_idx = gaussian_mixture(means, variances, weights, pair_n)
    return pair_data, pair_idx


class DiffusionModel(nn.Module):
    def __init__(self, n_io, n_hidden):
        super(DiffusionModel, self).__init__()
        self.value = torch.randn((K_BINS, K_BINS, K_BINS, 2)) / 1000

    def forward(self, a, neg_gamma):
        x, y = a[:, 0], a[:, 1]
        z = neg_gamma

        z_bins = torch.linspace(-11, 11, K_BINS+1)
        xy_bins = torch.linspace(-4.5, 4.5, K_BINS+1)

        x_idx = torch.searchsorted(xy_bins, x.unsqueeze(0)) - 1
        y_idx = torch.searchsorted(xy_bins, y.unsqueeze(0)) - 1
        z_idx = torch.searchsorted(z_bins, z.unsqueeze(0)) - 1

        return self.value[x_idx, y_idx, z_idx]


def f_neg_gamma(t, min_snr=-10, max_snr=10):
    return max_snr - t*(max_snr - min_snr)


def sigma_squared(neg_gamma):
    return torch.sigmoid(-neg_gamma)


def alpha_squared(neg_gamma):
    return torch.sigmoid(neg_gamma)


def diffusion_loss(model, data, f_neg_gamma):
    batch_size = data.shape[0]

    t = torch.rand((batch_size,))
    neg_gamma = f_neg_gamma(t)

    alpha, sigma = torch.sqrt(alpha_squared(neg_gamma)), torch.sqrt(sigma_squared(neg_gamma))
    epsilon = torch.randn_like(data)

    z = data*alpha[:, None] + sigma[:, None]*epsilon
    epsilon_hat = model(z, neg_gamma)

    neg_gamma_prime = torch.autograd.grad(neg_gamma.sum(), t, create_graph=True)[0]
    loss = -0.5 * neg_gamma_prime * (epsilon_hat.squeeze() - epsilon)**2

    return loss.sum()


def sample_diffusion(model, f_neg_gamma, n_steps, shape, n_samples):
    time_steps = torch.linspace(0, 1, n_steps+1)
    z = torch.randn(n_samples, *shape)

    for i in range(n_steps):
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]
        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)

        alpha_s = torch.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = torch.sqrt(alpha_squared(neg_gamma_t)), torch.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = model(z, neg_gamma_t)

        k = torch.exp((neg_gamma_t-neg_gamma_s)/2)
        z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat.squeeze()*(k-1))

    return z


def main():
    data, idx = make_data(MEANS, VARIANCES, WEIGHTS, PAIR_N)
    model = DiffusionModel(2, N_HIDDEN)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(B1, 0.999))

    for i in range(N_ITER):
        optimizer.zero_grad()
        loss = diffusion_loss(model, data, f_neg_gamma)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(i, loss.item())

    n_samples = 500
    n_steps = 100
    shape = (2,)
    samples = sample_diffusion(model, f_neg_gamma, n_steps, shape, n_samples)

    plt.scatter(samples[:, 0], samples[:, 1], c='r')
    plt.scatter(data[:, 0], data[:, 1], c='b')
    plt.show()


if __name__ == "__main__":
    main()
