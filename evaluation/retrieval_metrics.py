def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Calculate Precision@K.
    """

    if not isinstance(retrieved_ids, list):
        raise TypeError("Retrieved IDs must be a list.")

    if not isinstance(relevant_ids, set):
        raise TypeError("Relevant IDs must be a set.")

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("K must be an integer.")

    if k <= 0:
        raise ValueError("K must be greater than zero.")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for item in top_k if item in relevant_ids
    )

    return relevant_count / k


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Calculate Recall@K.
    """

    if not isinstance(retrieved_ids, list):
        raise TypeError("Retrieved IDs must be a list.")

    if not isinstance(relevant_ids, set):
        raise TypeError("Relevant IDs must be a set.")

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("K must be an integer.")

    if k <= 0:
        raise ValueError("K must be greater than zero.")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for item in top_k if item in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_results: list[list],
    relevant_ids: list[set]
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    """

    if not isinstance(retrieved_results, list):
        raise TypeError("Retrieved results must be a list.")

    if not isinstance(relevant_ids, list):
        raise TypeError("Relevant IDs must be a list.")

    if not retrieved_results:
        raise ValueError("Retrieved results cannot be empty.")

    if len(retrieved_results) != len(relevant_ids):
        raise ValueError(
            "Retrieved results and relevant IDs must have equal lengths."
        )

    reciprocal_ranks = []

    for results, relevant in zip(retrieved_results, relevant_ids):

        if not isinstance(results, list):
            raise TypeError("Each query result must be a list.")

        if not isinstance(relevant, set):
            raise TypeError("Each relevant ID collection must be a set.")

        if not relevant:
            raise ValueError("Relevant IDs cannot be empty.")

        reciprocal_rank = 0.0

        for rank, result_id in enumerate(results, start=1):

            if result_id in relevant:

                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)