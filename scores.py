import numpy as np
import logging

def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning with error handling."""
    if not strings_list:
        return []

    def are_equivalent(text1, text2):
        try:
            implication_1 = model.check_implication(text1, text2, example=example)
            implication_2 = model.check_implication(text2, text1, example=example)

            if not ((implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])):
                return False

            if strict_entailment:
                return (implication_1 == 2) and (implication_2 == 2)
            else:
                implications = [implication_1, implication_2]
                return (0 not in implications) and ([1, 1] != implications)

        except Exception as e:
            logging.error(f"Error in semantic comparison: {e}")
            return False

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    # Ensure all strings got an ID
    for i, id_val in enumerate(semantic_set_ids):
        if id_val == -1:
            semantic_set_ids[i] = next_id
            next_id += 1

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
    """Compute MC estimate of entropy with numerical stability."""
    if len(log_probs) == 0:
        return 0.0

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    entropy = -np.sum(log_probs) / (len(log_probs) + eps)

    # Clip to reasonable range
    return np.clip(entropy, -1e6, 1e6)


def predictive_entropy_rao(log_probs):
    """Compute Rao's quadratic entropy with numerical stability."""
    if len(log_probs) == 0:
        return 0.0

    # Normalize log probabilities for numerical stability
    max_log_prob = np.max(log_probs)
    log_probs_shifted = log_probs - max_log_prob

    # Convert to probabilities with numerical stability
    probs = np.exp(log_probs_shifted)
    probs = probs / (np.sum(probs) + 1e-10)

    # Calculate entropy
    entropy = -np.sum(probs * log_probs)
    return np.clip(entropy, -1e6, 1e6)


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

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = -(probabilities * np.log(probabilities)).sum()
    return entropy
