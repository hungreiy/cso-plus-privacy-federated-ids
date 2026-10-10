"""Ontology context for an alert: destination port -> service -> abused-for attack techniques (from CSO+)."""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ontology'))
from build_cso_plus import build

_triples, _inds = build()
_svc = defaultdict(list); _abused = defaultdict(list)
for s, p, o in _triples:
    if p == 'listensOn': _svc[o].append(s)
    if p == 'commonlyAbusedFor': _abused[s].append(o)

def path_for_port(port):
    node = f'Port_{port}'
    if node in _inds:
        svcs = _svc.get(node, [])
        tech = sorted(set(_abused.get(node, [])))
        if svcs:
            return f"{node} -> {'/'.join(svcs)} -> " + (', '.join(tech) if tech else 'no attack technique listed')
        return f'{node} -> WellKnownRange (no service or attack technique listed)'
    return f'Port_{port} -> ' + ('RegisteredRange' if port <= 49151 else 'DynamicRange') + ' (no service or attack technique listed)'
