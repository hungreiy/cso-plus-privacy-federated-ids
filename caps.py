"""Participation cap for federated rounds (no Flower import, so it can be unit-tested).
A tenant that has already trained `cap` rounds is not asked to train again. The rule does not depend on any tenant's data,
so the privacy accounting stays valid, and it makes the busiest tenant's epsilon at most the epsilon of `cap` rounds."""
def apply_cap(instructions, counts, cap):
    """instructions: list of (client, fit_ins) with client.cid; counts: dict cid -> rounds trained so far (updated in place)."""
    if not cap: return instructions
    keep = []
    for client, ins in instructions:
        c = counts.get(client.cid, 0)
        if c < cap:
            counts[client.cid] = c + 1
            keep.append((client, ins))
    return keep
