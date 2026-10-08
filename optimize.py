from contextlib import contextmanager
from copy import deepcopy

import torch
from sklearn.metrics import roc_auc_score
from torch.func import functional_call

from model import predict


def ranking_loss(model, graph, batch, parameters=None):
    """Summed source-conditioned ranking loss, Eqs. (25)--(27)."""
    positive, negative = batch
    same_source = positive[:, :1] == negative[:, 0].unsqueeze(0)
    if not len(positive) or not same_source.any(dim=1).all():
        raise ValueError("Every positive source needs at least one negative target.")
    parameters = dict(model.named_parameters()) if parameters is None else parameters
    scores = functional_call(model, parameters, (*graph, torch.cat(batch)),
                             {"hard": False}, strict=True)
    pos, neg = scores[:len(positive)], scores[len(positive):]
    neg = neg.unsqueeze(0).expand(len(pos), -1).masked_fill(~same_source, -torch.inf)
    return (torch.logsumexp(torch.cat((pos[:, None], neg), dim=1), dim=1) - pos).sum()


@contextmanager
def fixed_randomness(model, seed):
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


def selector_gradient(model, graph, train_batch, valid_batch, virtual_lr,
                      epsilon, train_seed, valid_seed):
    if virtual_lr <= 0 or epsilon <= 0:
        raise ValueError("virtual_lr and epsilon must be positive.")
    parameters = dict(model.named_parameters())
    theta = {name: value for name, value in parameters.items()
             if not name.startswith("selector.")}
    omega = tuple(model.selector.parameters())

    def train_loss(values):
        with fixed_randomness(model, train_seed):
            return ranking_loss(model, graph, train_batch, values)

    # Eq. (29): temporary parameters are separate leaves, not an Adam step.
    gradient = torch.autograd.grad(train_loss(parameters), tuple(theta.values()))
    temporary = {name: (value - virtual_lr * grad).detach().requires_grad_()
                 for (name, value), grad in zip(theta.items(), gradient)}
    with fixed_randomness(model, valid_seed):
        validation = ranking_loss(model, graph, valid_batch, {**parameters, **temporary})
    partials = torch.autograd.grad(validation, tuple(temporary.values()) + omega)
    direction = [value.detach() for value in partials[:len(theta)]]
    direct = partials[len(theta):]

    # Eqs. (32)--(34): center at theta and reuse the training randomness.
    shifted = []
    for sign in (1, -1):
        point = {name: value.detach() + sign * epsilon * vector
                 for (name, value), vector in zip(theta.items(), direction)}
        loss = train_loss({**parameters, **point})
        shifted.append(torch.autograd.grad(loss, omega))
    return tuple((g - virtual_lr * (plus - minus) / (2 * epsilon)).detach()
                 for g, plus, minus in zip(direct, *shifted))


def bilevel_step(model, graph, train_batch, valid_batch,
                 theta_optimizer, omega_optimizer, epsilon, grad_clip):
    model.train()
    train_seed, valid_seed = torch.randint(2**31 - 1, (2,)).tolist()
    virtual_lr = theta_optimizer.param_groups[0]["lr"]
    hypergradient = selector_gradient(model, graph, train_batch, valid_batch,
                                     virtual_lr, epsilon, train_seed, valid_seed)
    omega_optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(model.selector.parameters(), hypergradient):
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(model.selector.parameters(), grad_clip,
                                  error_if_nonfinite=True)
    omega_optimizer.step()

    # The updated selector is fixed; only theta receives the training update.
    theta = [value for name, value in model.named_parameters()
             if not name.startswith("selector.")]
    theta_optimizer.zero_grad(set_to_none=True)
    with fixed_randomness(model, train_seed):
        loss = ranking_loss(model, graph, train_batch)
    for parameter, gradient in zip(theta, torch.autograd.grad(loss, theta)):
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(theta, grad_clip, error_if_nonfinite=True)
    theta_optimizer.step()
    return float(loss.detach())


def batch_from_pool(positive, negative_pool):
    selected = torch.isin(negative_pool[:, 0], positive[:, 0])
    return positive, negative_pool[selected]


def fit(model, graph, train_positive, train_negative, valid_positive,
        valid_negative, config):
    theta = [value for name, value in model.named_parameters()
             if not name.startswith("selector.")]
    theta_optimizer = torch.optim.Adam(theta, lr=config["lr"])
    omega_optimizer = torch.optim.Adam(model.selector.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        theta_optimizer, T_max=config["epochs"], eta_min=config["lr_min"])
    valid_pairs = torch.cat((valid_positive, valid_negative))
    labels = [1] * len(valid_positive) + [0] * len(valid_negative)
    best_auc, best_state, stale = -float("inf"), None, 0
    for _ in range(config["epochs"]):
        order = torch.randperm(len(train_positive), device=train_positive.device)
        for indices in order.split(config["batch_size"]):
            valid_indices = torch.randperm(len(valid_positive), device=valid_positive.device)
            train_batch = batch_from_pool(train_positive[indices], train_negative)
            valid_batch = batch_from_pool(
                valid_positive[valid_indices[:config["batch_size"]]], valid_negative)
            bilevel_step(model, graph, train_batch, valid_batch, theta_optimizer,
                         omega_optimizer, config["epsilon"], config["grad_clip"])
        scheduler.step()
        probability = predict(model, graph, valid_pairs, config["batch_size"])
        auc = roc_auc_score(labels, probability.cpu().numpy())
        if auc > best_auc:
            best_auc, best_state, stale = auc, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return float(best_auc)
