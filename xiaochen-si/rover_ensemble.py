"""Pure Python ROVER (Recognizer Output Voting Error Reduction) implementation.

Combines outputs from multiple ASR systems via word-level alignment and voting.
No external C libraries required.

Reference: Fiscus, J.G. (1997). "A post-processing system to yield reduced word
error rates: Recognizer Output Voting Error Reduction (ROVER)"

Usage:
    from rover_ensemble import rover_combine
    result = rover_combine(["hello world", "hello word", "helo world"])
    # result = "hello world"
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Word-level Levenshtein alignment
# ---------------------------------------------------------------------------

def align_two_sequences(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Align two word sequences using Levenshtein distance.

    Returns list of (ref_word, hyp_word) pairs.
    None indicates an insertion or deletion.
    """
    n, m = len(ref), len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrace
    aligned: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            aligned.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # substitution
            aligned.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # deletion from ref
            aligned.append((ref[i - 1], None))
            i -= 1
        else:
            # insertion from hyp
            aligned.append((None, hyp[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


# ---------------------------------------------------------------------------
# Word Transition Network (WTN)
# ---------------------------------------------------------------------------

@dataclass
class WTNArc:
    """An arc in the Word Transition Network."""
    word: str | None  # None = epsilon (deletion)
    votes: float = 1.0


@dataclass
class WTNNode:
    """A node/slot in the Word Transition Network."""
    arcs: list[WTNArc] = field(default_factory=list)

    def add_vote(self, word: str | None, weight: float = 1.0):
        """Add a vote for a word at this slot."""
        for arc in self.arcs:
            if arc.word == word:
                arc.votes += weight
                return
        self.arcs.append(WTNArc(word=word, votes=weight))

    def best_word(self) -> str | None:
        """Return the word with the most votes. None = deletion."""
        if not self.arcs:
            return None
        return max(self.arcs, key=lambda a: a.votes).word


def build_wtn_from_two(
    wtn_words: list[str | None],
    new_hyp: list[str],
    weight: float = 1.0,
) -> list[WTNNode]:
    """Align a new hypothesis against the current WTN backbone and merge.

    The WTN backbone is represented as a list of words (from the current best path).
    Returns updated WTN nodes.
    """
    # Filter out Nones for alignment purposes
    backbone = [w for w in wtn_words if w is not None]
    alignment = align_two_sequences(backbone, new_hyp)

    # Map alignment back to WTN nodes
    nodes: list[WTNNode] = []
    for ref_word, hyp_word in alignment:
        node = WTNNode()
        if ref_word is not None:
            node.add_vote(ref_word, weight=1.0)  # existing vote
        else:
            node.add_vote(None, weight=1.0)  # existing epsilon
        node.add_vote(hyp_word, weight=weight)
        nodes.append(node)

    return nodes


# ---------------------------------------------------------------------------
# ROVER Main Function
# ---------------------------------------------------------------------------

def rover_combine(
    hypotheses: list[str],
    weights: list[float] | None = None,
) -> str:
    """Combine multiple ASR hypotheses using ROVER.

    Args:
        hypotheses: List of transcript strings from different ASR systems.
        weights: Optional confidence weights for each system (default: equal).

    Returns:
        The ROVER-combined transcript string.
    """
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]

    n_sys = len(hypotheses)
    if weights is None:
        weights = [1.0] * n_sys

    # Tokenize
    token_lists = [h.split() for h in hypotheses]

    # Handle edge case: all empty
    if all(len(t) == 0 for t in token_lists):
        return ""

    # Start with the first hypothesis as the backbone
    # (choosing the longest as backbone often gives better alignment)
    backbone_idx = max(range(n_sys), key=lambda i: len(token_lists[i]))
    backbone = token_lists[backbone_idx]

    # Initialize WTN from backbone
    nodes: list[WTNNode] = []
    for word in backbone:
        node = WTNNode()
        node.add_vote(word, weight=weights[backbone_idx])
        nodes.append(node)

    # Incrementally align each other hypothesis
    for i in range(n_sys):
        if i == backbone_idx:
            continue

        hyp = token_lists[i]
        w = weights[i]

        if not hyp:
            # This system produced empty output - add epsilon votes to all nodes
            for node in nodes:
                node.add_vote(None, weight=w)
            continue

        # Get current backbone words for alignment
        current_backbone = []
        for node in nodes:
            best = node.best_word()
            if best is not None:
                current_backbone.append(best)

        # Align new hyp against current backbone
        alignment = align_two_sequences(current_backbone, hyp)

        # Merge alignment into existing WTN
        new_nodes: list[WTNNode] = []
        backbone_ptr = 0  # pointer into existing nodes (skipping epsilon-only nodes)
        node_ptr = 0  # pointer into all existing nodes

        for ref_word, hyp_word in alignment:
            if ref_word is not None:
                # Find the next non-epsilon node
                while node_ptr < len(nodes) and nodes[node_ptr].best_word() is None:
                    new_nodes.append(nodes[node_ptr])
                    node_ptr += 1
                if node_ptr < len(nodes):
                    nodes[node_ptr].add_vote(hyp_word, weight=w)
                    new_nodes.append(nodes[node_ptr])
                    node_ptr += 1
                else:
                    # Shouldn't happen, but handle gracefully
                    node = WTNNode()
                    node.add_vote(ref_word, weight=1.0)
                    node.add_vote(hyp_word, weight=w)
                    new_nodes.append(node)
            else:
                # Insertion: hyp has a word not in backbone
                node = WTNNode()
                node.add_vote(None, weight=1.0)  # other systems vote for deletion
                node.add_vote(hyp_word, weight=w)
                new_nodes.append(node)

        # Add remaining nodes
        while node_ptr < len(nodes):
            nodes[node_ptr].add_vote(None, weight=w)
            new_nodes.append(nodes[node_ptr])
            node_ptr += 1

        nodes = new_nodes

    # Extract best path
    result_words = []
    for node in nodes:
        best = node.best_word()
        if best is not None:
            result_words.append(best)

    return " ".join(result_words)


# ---------------------------------------------------------------------------
# Batch ROVER
# ---------------------------------------------------------------------------

def rover_combine_batch(
    hypotheses_list: list[list[str]],
    weights: list[float] | None = None,
) -> list[str]:
    """Apply ROVER to a batch of utterances.

    Args:
        hypotheses_list: For each utterance, a list of hypotheses from different systems.
            Shape: [n_utterances][n_systems]
        weights: Optional confidence weights per system.

    Returns:
        List of combined transcripts, one per utterance.
    """
    results = []
    for hyps in hypotheses_list:
        results.append(rover_combine(hyps, weights))
    return results


# ---------------------------------------------------------------------------
# Simple majority voting (faster alternative)
# ---------------------------------------------------------------------------

def majority_vote(hypotheses: list[str], weights: list[float] | None = None) -> str:
    """Simple word-level majority voting via pairwise alignment.

    Faster than full ROVER but less accurate for very different outputs.
    Falls back to the hypothesis with highest weight if all are different.
    """
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]

    # If two systems agree exactly, return that
    from collections import Counter
    counts = Counter(hypotheses)
    most_common = counts.most_common(1)[0]
    if most_common[1] > 1:
        return most_common[0]

    # Otherwise use full ROVER
    return rover_combine(hypotheses, weights)


if __name__ == "__main__":
    # Quick test
    hyps = [
        "the cat sat on the mat",
        "the cat sit on the mat",
        "a cat sat on the mat",
    ]
    result = rover_combine(hyps)
    print(f"ROVER result: '{result}'")

    hyps2 = [
        "hello world",
        "hello word",
        "helo world",
    ]
    result2 = rover_combine(hyps2)
    print(f"ROVER result: '{result2}'")

    hyps3 = [
        "i goed to the store",
        "i go to the store",
        "i goed to the store",
    ]
    result3 = rover_combine(hyps3)
    print(f"ROVER result: '{result3}'")
