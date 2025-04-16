import torch
import random
import numpy as np
from tqdm import tqdm

def seed_everything(seed):
    """
    Set the seed for reproducibility across various libraries.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def topk_lowest_masking(scores, cutoff_len):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    sorted_index = scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    return scores < cutoff

def stochastic_sample_from_categorical(logits, temperature=1.0, noise_scale=1.0):
    """
    Sample from a categorical distribution with optional temperature scaling and Gumbel noise.
    """
    logits = logits.double()
    if temperature != 0:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits / temperature + noise_scale * gumbel_noise
    scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

@torch.inference_mode()
@torch.cuda.amp.autocast()
def path_planning_sampler(
    xt, model, tokenizer, num_steps,
    tau=1.0, kappa_fn=lambda t: t, eta=1, alpha=1.,
    planner=None, score_type='confidence'
): 
    """
    Stochastic remasking sampling method for iterative refinement of sequences.

    Args:
        xt (Tensor): Initial token tensor.
        model (callable): Model function mapping tokens to logits.
        tokenizer (object): Tokenizer with `mask_token_id` attribute.
        num_steps (int): Number of refinement steps.
        tau (float): Temperature parameter for softmax sampling.
        kappa_fn (callable): Function controlling the unmasking schedule.
        eta (float): Scaling factor for score adjustments.
        alpha (float): Weighting for confidence-based scoring.
        planner (callable, optional): Additional model for planning logits.
        score_type (str): Scoring method, 'confidence' or 'random'.

    Returns:
        Tensor: Final sampled sequence tensor.
    """
    assert score_type in ['confidence', 'random']
    
    dt = 1 / num_steps
    fix_mask = xt != tokenizer.mask_token_id

    for i in tqdm(range(1, num_steps + 1)):
        kappa_t = kappa_fn(i * dt)
        logits = model(xt).double()
        last_mask = xt == tokenizer.mask_token_id
        unmask_t = ~last_mask & ~fix_mask

        x0, logp = stochastic_sample_from_categorical(logits, temperature=tau)

        if planner is not None:
            planner_logits = planner(x0).double()
            planner_logp = planner_logits.log_softmax(dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            logits[unmask_t] = planner_logits[unmask_t]
            logp[unmask_t] = planner_logp[unmask_t]

        if score_type == 'confidence':
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            score = alpha * logp + (1 - alpha) * -entropy
            score = score.masked_fill(fix_mask, float('inf'))
        elif score_type == 'random':
            score = torch.rand_like(logp).log()
            score = score.masked_fill(fix_mask, float('inf'))
        else:
            raise ValueError(f"Invalid score type: {score_type}")

        score[unmask_t] = score[unmask_t] * eta

        num_to_mask = ((~fix_mask).sum(1, keepdim=True).float() * (1 - kappa_t)).long()
        lowest_k_mask = topk_lowest_masking(score, num_to_mask)

        xt[lowest_k_mask] = tokenizer.mask_token_id
        mask_2_x0 = last_mask & ~lowest_k_mask
        xt[mask_2_x0] = x0[mask_2_x0]

        print(f"Step {i}/{num_steps} | eta: {eta}, alpha: {alpha}, Stochastic remask: \n", xt[0])

    xt[xt == tokenizer.mask_token_id] = x0[xt == tokenizer.mask_token_id]
    return xt