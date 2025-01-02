import numpy as np
import logging
import torch

def context_entails_response(context, responses, model):
    """
    Returns a score between 0 and 1 representing how strongly the context entails the responses.
    Uses full probability distribution from DeBERTa.
    """
    all_scores = []
    for response in responses:
        probs = model.check_implication(context, response)
        score = (
            -0.5 * probs["contradiction"]  # Penalize contradictions
            + 0.2 * probs["neutral"]       # Less credit for neutral
            + 1.0 * probs["entailment"]    # Full credit for entailment
        )
        # Clip to [0,1] range after applying weights
        score = np.clip(score, 0, 1)
        all_scores.append(score)
    return np.mean(all_scores)

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    logging.info(f"\nCalculating semantic IDs for {len(strings_list)} solutions")

    if not strings_list:
        logging.warning("Empty strings list provided to get_semantic_ids")
        return []

    def are_equivalent(text1, text2):
        probs_1_to_2 = model.check_implication(text1, text2)
        logging.info(f"Probs 1 to 2: {probs_1_to_2}")
        probs_2_to_1 = model.check_implication(text2, text1)
        logging.info(f"Probs 2 to 1: {probs_2_to_1}")
        
        # Stricter thresholds
        entailment_threshold = 0.8
        contradiction_threshold = 0.2
        neutral_threshold = 0.5
        
        # Require bidirectional entailment and low contradiction
        is_equivalent = (
            probs_1_to_2["entailment"] > entailment_threshold
            and probs_2_to_1["entailment"] > entailment_threshold
            and probs_1_to_2["contradiction"] < contradiction_threshold
            and probs_2_to_1["contradiction"] < contradiction_threshold
            and probs_1_to_2["neutral"] < neutral_threshold
            and probs_2_to_1["neutral"] < neutral_threshold
        )
        
        return is_equivalent

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            group_size = 1
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
                    group_size += 1
            logging.info(f"Created semantic group {next_id} with {group_size} members")
            next_id += 1

    # Log distribution of semantic IDs
    unique_ids = set(semantic_set_ids)
    id_counts = {id_val: semantic_set_ids.count(id_val) for id_val in unique_ids}
    logging.info(f"Final semantic ID distribution: {id_counts}")

    return semantic_set_ids



def logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized"):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == "sum_normalized":
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy with improved numerical stability."""
    if len(log_probs) == 0:
        return 0.0

    # Convert log probabilities to probabilities with numerical stability
    max_log_prob = np.max(log_probs)
    log_probs_shifted = log_probs - max_log_prob
    probs = np.exp(log_probs_shifted)
    probs = probs / (np.sum(probs) + 1e-10)  # Add epsilon to avoid division by zero

    # Calculate entropy using probabilities
    entropy = -np.sum(probs * log_probs_shifted)

    # Add small epsilon to handle edge cases
    eps = 1e-10
    entropy = entropy + eps

    return np.clip(entropy, 0, 1e3)


def predictive_entropy_rao(log_probs):
    """Compute Rao's quadratic entropy with improved numerical stability."""
    if len(log_probs) == 0:
        return 0.0

    # Normalize log probabilities with numerical stability
    max_log_prob = np.max(log_probs)
    log_probs_shifted = log_probs - max_log_prob
    probs = np.exp(log_probs_shifted)
    probs = probs / (np.sum(probs) + 1e-10)  # Add epsilon to avoid division by zero

    # Calculate Rao's entropy with protection against nan/inf
    entropy = 0.0
    eps = 1e-10

    for i, p_i in enumerate(probs):
        for j, p_j in enumerate(probs):
            if i != j:
                d_ij = abs(log_probs[i] - log_probs[j])  # Distance measure
                entropy += p_i * p_j * d_ij

    entropy = entropy / 2.0 + eps  # Add small epsilon for stability
    return np.clip(entropy, 0, 1e3)


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """
    logging.info("\nCalculating cluster assignment entropy")
    logging.debug(f"Semantic IDs: {semantic_ids}")

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations

    if not np.isclose(probabilities.sum(), 1):
        logging.error(f"Probability sum error: {probabilities.sum()}")

    entropy = -(probabilities * np.log(probabilities)).sum()
    return entropy
