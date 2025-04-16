import argparse
import os
from pprint import pprint
import math
from faesm.model.esm import FAEsmForMaskedLM
from faesm.scripts.generate import path_planning_sampler, seed_everything
import torch
import time
from functools import partial


def compute_foldability(df):
    """
    Compute foldability percentage for an entire DataFrame. 
    Foldable if pLDDT > 80, pTM > 0.7, pAE < 10.

    Args:
        df (pd.DataFrame): DataFrame containing sequences and metrics.

    Returns:
        float: Foldability percentage.
    """
    if df.empty:
        return 0.0
    foldable_count = df[
        (df['pLDDT'] > 80) & (df['pTM'] > 0.7) & (df['pAE'] < 10)
    ].shape[0]
    total_count = df.shape[0]
    return (foldable_count / total_count) * 100 if total_count > 0 else 0


def parse_prompt(
    prompt, 
    B, 
    tokenizer, 
    device, 
    seq_len, 
    add_special_tokens=True
):
    """
    A merged function that either:
    1) Parses the input `prompt` if it is non-empty and not purely numeric, 
       building a protein sequence according to the mini-language:
         - <mask>*N => expand <mask> N times
         - SEQ*N    => expand literal SEQ N times
         - chunks separated by '+'
    2) If the prompt is empty or purely numeric, constructs a default sequence
       of length `seq_len` using <mask> tokens.

    Args:
        prompt (str): The protein sequence template or an empty/numeric string.
        B (int): Batch size.
        tokenizer: The tokenizer to convert text to token IDs.
        device: The device to place the final tensor (e.g., 'cuda' or 'cpu').
        seq_len (int): The length of the sequence if we default to <mask>.
        add_special_tokens (bool): Whether to include special tokens.

    Returns:
        torch.Tensor: A tokenized batch of shape [B, L].
    """

    # Check if prompt is empty or purely numeric
    prompt_stripped = prompt.strip() if prompt is not None else ""
    is_pure_numeric = prompt_stripped.isdigit()
    is_empty = (prompt_stripped == "")

    if is_empty or is_pure_numeric:
        # --- Default sequence of <mask> tokens ---
        # 1) Construct the single sequence of <mask> repeated seq_len times
        seq = ["<mask>"] * seq_len
        final_sequence = "".join(seq)
    else:
        # --- Parse the prompt ---
        chunks = prompt_stripped.split('+')
        expanded_sequence = []

        for chunk in chunks:
            if '*' in chunk:
                base, count_str = chunk.split('*')
                count = int(count_str)
                expanded_sequence.append(base * count)
            else:
                expanded_sequence.append(chunk)

        final_sequence = "".join(expanded_sequence)

    batch_sequences = [final_sequence for _ in range(B)]

    tokenized = tokenizer.batch_encode_plus(
        batch_sequences,
        add_special_tokens=add_special_tokens,
        padding='longest',
        return_tensors='pt'
    )

    input_ids = tokenized["input_ids"].to(device)
    
    return input_ids

def get_initial(num_seqs, tokenizer, device, seq_len, add_special_tokens=True):
    """
    Prepare the initial input_ids for the sampling routine.
    
    Args:
        num_seqs (int): Number of sequences to generate.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        device (torch.device): The device to place tensors on.
        seq_len (int): The length of the sequence to generate.
        add_special_tokens (bool): Whether to include special tokens.
    
    Returns:
        Tensor: The initial input_ids tensor of shape [num_seqs, L].
    """
    seq = ['<mask>'] * seq_len
    initial_sequence = ''.join(seq)
    init_seqs = [initial_sequence] * num_seqs
    batch = tokenizer.batch_encode_plus(
        init_seqs,
        add_special_tokens=add_special_tokens,
        padding="longest",
        return_tensors='pt'
    )["input_ids"].to(device)
    return batch

def ignore_special_tokens_logits(logits, tokenizer):
    """
    Masks out the logits of special tokens to prevent them from being sampled.
    
    Args:
        logits (Tensor): Logits output from the model of shape [B, L, V].
        tokenizer: The tokenizer to access special token IDs.
    
    Returns:
        Tensor: Modified logits with special tokens masked out.
    """
    logits[..., tokenizer.mask_token_id] = -math.inf
    logits[..., tokenizer._token_to_id["X"]] = -math.inf
    logits[..., tokenizer.pad_token_id] = -math.inf
    logits[..., tokenizer.cls_token_id] = -math.inf
    logits[..., tokenizer.eos_token_id] = -math.inf
    return logits

def parse_arguments():
    """
    Parse command line arguments and return the argparse.Namespace object.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Protein Sequence Generation and Evaluation")

    # General Settings
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--model_name', type=str, default='zhangzhi/EvoFlow-650M-context-3070', help="Pretrained model name.")
    parser.add_argument('--num_seqs', type=int, default=100, help="Number of sequences to generate.")
    parser.add_argument('--num_steps', type=int, default=128, help="Number of sampling steps (<= sequence length for ancestral sampling).")
    parser.add_argument('--sampling_alg', type=str, default='ancestral',  help="Sampling algorithm to use.")
    parser.add_argument('--corrector_name', type=str, default=None, help="Name of the corrector model for 'corrector' sampling algorithm.")
    parser.add_argument('--seq_lens', nargs='*', type=int, default=[128], help="List of sequence lengths to generate.")
    parser.add_argument('--saveto', type=str, default='generation-results/ancestral', help="Directory to save generated sequences.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature parameter for sampling.")
    parser.add_argument('--max_tokens_per_batch', type=int, default=10000, help="Maximum tokens per batch for ESMFold evaluation.")
    parser.add_argument('--no_esm_eval', action='store_true', help="Disable ESMFold evaluation.")
    parser.add_argument('--prompt', type=str, default='', help="Prompt defines the template sequence with <mask>s to be generated.")
    parser.add_argument('--eta', type=float, default=1.0, 
                        help="stochasticity strength. if eta -> 0, it's greedy ancestral, ")
    parser.add_argument('--alpha', type=float, default=1.0, help="weight terms for scores")
    # add score
    parser.add_argument('--score_type', type=str, default='confidence', help="score for ranking positions")
    args = parser.parse_args()
    return args
def run(
    seed=None,
    model_name='airkingbd/dplm_150m',
    num_seqs=100,
    num_steps=128,
    corrector_name=None,
    seq_lens=[128],
    saveto='generation-results/ancestral',
    temperature=1.0,
    max_tokens_per_batch=10000,
    no_esm_eval=False,
    prompt='',
    eta=1.0,
    alpha=1.0,
    score_type='confidence',
    device='cuda', 
):
    """
    Run the sequence generation and evaluation.

    Args:
        seed (int, optional): Random seed for reproducibility.
        model_name (str): Pretrained model name.
        num_seqs (int): Number of sequences to generate.
        num_steps (int): Number of sampling steps.
        sampling_alg (str): Sampling algorithm to use ('ancestral', 'remasking', 'corrector').
        corrector_name (str, optional): Name of the corrector model if using 'corrector' sampling algorithm.
        seq_lens (list of int): List of sequence lengths to generate.
        saveto (str): Directory to save generated sequences.
        temperature (float): Temperature parameter for sampling.
        max_tokens_per_batch (int): Maximum tokens per batch for ESMFold evaluation.
        no_esm_eval (bool): If True, skip ESMFold evaluation.
        prompt (str): Template sequence with <mask>s to be generated.
        eta (float): Hyperparameter for stochastic remasking.
        alpha (float): Scaling factor for some algorithms.
        score_type (str): Scoring type for sequence evaluation.
        device (str): Device to run the model on (e.g., 'cuda', 'cpu').

    Returns:
        dict: A dictionary mapping each sequence length to its sampling elapsed time in seconds.

    Raises:
        ValueError: If an invalid sampling algorithm is provided or required parameters are missing.
    """
    # Set seed for reproducibility
    seed_everything(seed)
    print(f"Using denoiser model: {model_name}")

    # Load the denoiser model and move it to the specified device
    model = FAEsmForMaskedLM.from_pretrained(model_name)
    tokenizer = model.tokenizer
    model = model.eval().to(device)

    # Define the model wrapper function
    def model_wrapper(x):
        out = ignore_special_tokens_logits(model(x.to(device))['logits'].float(), tokenizer)
        return out

    # Create the directory to save all FASTA files
    base_save_dir = saveto
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Select the sampling function based on the sampling algorithm
    
    if corrector_name is not None and corrector_name != '' and corrector_name != 'None':
        corrector = FAEsmForMaskedLM.from_pretrained(corrector_name).eval().to(device)
        corrector_wrapper = lambda x: ignore_special_tokens_logits(corrector(x.to(device))['logits'].float(), tokenizer)
    else:
        corrector_wrapper = None
    sample_fn = partial(
        path_planning_sampler,
        eta=eta,
        alpha=alpha,
        planner=corrector_wrapper,
        score_type=score_type,
    )
    
    # Dictionary to store sampling times for each sequence length
    sampling_times = {}

    # Iterate over each sequence length
    for seq_len in seq_lens:
        print(f"Generating sequences of length: {seq_len}")

        # Prepare initial tokens using the prompt
        xt = parse_prompt(prompt, num_seqs, tokenizer, device, seq_len)
        print(f"xt: {xt.shape}")
        

        # Start timing the sampling process
        start_time = time.time()

        # Perform sampling
        sampled_xt = sample_fn(
            xt=xt,
            model=model_wrapper,
            tokenizer=tokenizer,
            num_steps=num_steps,
            tau=temperature,    # Using temperature as tau
            kappa_fn=lambda t: t  # Linear schedule; can be customized
        )

        # End timing the sampling process
        end_time = time.time()
        elapsed_time = end_time - start_time
        sampling_times[seq_len] = elapsed_time

        # Decode the sampled sequences
        decoded_seqs = tokenizer.batch_decode(sampled_xt, skip_special_tokens=True)
        decoded_seqs = [''.join(seq.split()) for seq in decoded_seqs]
        pprint(decoded_seqs)
        decoded_seqs = list(set(decoded_seqs))
        print(f"Number of unique sequences: {len(decoded_seqs)}")

        # Save the sequences to a FASTA file
        saveto_name = os.path.join(base_save_dir, f"L_{seq_len}.fasta")
        with open(saveto_name, 'w') as fp_save:
            for idx, seq in enumerate(decoded_seqs):
                fp_save.write(f">SEQUENCE_{idx}_L={seq_len}\n")
                fp_save.write(f"{seq}\n")
        print(f"Saved sequences to {saveto_name}\n")

    # Optional ESMFold evaluation
    if not no_esm_eval:
        from analysis.cal_plddt_dir import run as run_esmfold_eval
        from pathlib import Path
        import pandas as pd

        print("Starting ESMFold evaluation...")
        df = run_esmfold_eval(
            Path(base_save_dir),
            Path(base_save_dir).joinpath("esmfold_pdb"),
            num_recycles=1,
            max_tokens_per_batch=max_tokens_per_batch,
            chunk_size=None,
            cpu_only=(device == 'cpu'),
            cpu_offload=False,
        )
        df['Sampling Step'] = num_steps
        df['Temperature'] = temperature
        df['Model Name'] = model_name
        df['eta'] = eta
        df['alpha'] = alpha
        df['Planner'] = corrector_name
        df['score_type'] = score_type
        df['prompt'] = prompt
        run_time = sum(sampling_times.values())
        df['Elapsed Time (s)'] = run_time
        df['Token/sec'] = round(num_seqs * seq_len / run_time, 2)
        df_path = Path(base_save_dir).joinpath("esmfold_results.csv")
        df.to_csv(df_path, index=False)
        print(f"Saved ESMFold results to {df_path}")
        foldability = compute_foldability(df)
        print(f"Foldability: {foldability:.2f}%")
    return sampling_times


def main():
    """
    Main function that ties together argument parsing and running the experiment.
    """
    args = parse_arguments()
    sampling_times = run(
        seed=args.seed,
        model_name=args.model_name,
        num_seqs=args.num_seqs,
        num_steps=args.num_steps,
        corrector_name=args.corrector_name,
        seq_lens=args.seq_lens,
        saveto=args.saveto,
        temperature=args.temperature,
        max_tokens_per_batch=args.max_tokens_per_batch,
        no_esm_eval=args.no_esm_eval,
        prompt=args.prompt,
        eta=args.eta,
        alpha=args.alpha,
        score_type=args.score_type,
    )
    # Optionally, you can print or log the sampling times here
    print("Sampling Times (in seconds) per Sequence Length:")
    pprint(sampling_times)

if __name__ == '__main__':
    main()
    
