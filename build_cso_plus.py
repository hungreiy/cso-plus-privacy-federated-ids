"""Build CSO+ (a cloud-security ontology written for this thesis), export it as Turtle,
derive the knowledge graph, learn Node2Vec embeddings (NumPy) and write a port -> vector table.

Outputs (in --out_dir):
  cso_plus.ttl              ontology + individuals in Turtle (open in Protege to run a reasoner)
  cso_plus_node2vec.npz     `ports` (0..65535) and `vectors` (65536 x 32) -> data.py --kg
  cso_plus_stats.json       class / property / individual / triple / node / edge counts
  cso_plus_classes.json     classes grouped by theme (for the thesis text)

The relations below are general security knowledge written by the author. They were written
knowing the attack types in CIC-IDS2017, which is a threat to validity (state it in Chapter 4).
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

NS = 'http://example.org/csoplus#'

# ---------------- classes: (name, parent, theme) ----------------
CLASSES = [
 ('CloudAsset', None, 'core'), ('Threat', None, 'core'), ('AttackTechnique', 'Threat', 'core'),
 ('NetworkEntity', None, 'core'), ('Control', None, 'core'), ('Tenant', None, 'core'),
 ('Port', 'NetworkEntity', 'network'), ('PortRange', 'NetworkEntity', 'network'),
 ('Protocol', 'NetworkEntity', 'network'), ('Service', 'NetworkEntity', 'network'),
 # containerisation
 ('Container', 'CloudAsset', 'containerisation'), ('Pod', 'CloudAsset', 'containerisation'),
 ('Deployment', 'CloudAsset', 'containerisation'), ('DaemonSet', 'CloudAsset', 'containerisation'),
 ('Sidecar', 'Container', 'containerisation'), ('ContainerRegistry', 'CloudAsset', 'containerisation'),
 ('KubernetesAPIServer', 'Service', 'containerisation'), ('Kubelet', 'Service', 'containerisation'),
 # serverless and orchestration
 ('Function', 'CloudAsset', 'serverless'), ('FunctionInvocation', 'CloudAsset', 'serverless'),
 ('EventTrigger', 'CloudAsset', 'serverless'), ('WorkflowOrchestrator', 'CloudAsset', 'serverless'),
 ('APIGateway', 'Service', 'serverless'), ('ObjectStorageBucket', 'CloudAsset', 'serverless'),
 # service mesh and networking
 ('ServiceMesh', 'CloudAsset', 'mesh'), ('VirtualService', 'CloudAsset', 'mesh'),
 ('IngressController', 'CloudAsset', 'mesh'), ('NetworkPolicy', 'Control', 'mesh'),
 ('LoadBalancer', 'CloudAsset', 'mesh'), ('VirtualNetwork', 'CloudAsset', 'mesh'),
 # identity and multi-tenancy
 ('IAMRole', 'CloudAsset', 'identity'), ('ServiceAccount', 'CloudAsset', 'identity'),
 ('TenantBoundary', 'Control', 'identity'), ('CrossTenantFlow', 'CloudAsset', 'identity'),
 ('AccessKey', 'CloudAsset', 'identity'),
 # cloud-native attack techniques
 ('ContainerEscape', 'AttackTechnique', 'cloud_attacks'), ('SupplyChainCompromise', 'AttackTechnique', 'cloud_attacks'),
 ('PrivilegeEscalation', 'AttackTechnique', 'cloud_attacks'), ('Cryptojacking', 'AttackTechnique', 'cloud_attacks'),
 ('LateralMovement', 'AttackTechnique', 'cloud_attacks'), ('DataExfiltration', 'AttackTechnique', 'cloud_attacks'),
 ('ExposedAPIExploit', 'AttackTechnique', 'cloud_attacks'),
 # general network attack techniques
 ('Reconnaissance', 'AttackTechnique', 'network_attacks'), ('BruteForce', 'AttackTechnique', 'network_attacks'),
 ('DenialOfService', 'AttackTechnique', 'network_attacks'), ('WebAttack', 'AttackTechnique', 'network_attacks'),
 ('CommandAndControl', 'AttackTechnique', 'network_attacks'), ('ProtocolExploit', 'AttackTechnique', 'network_attacks'),
 ('DataExposure', 'AttackTechnique', 'network_attacks'),
]
# ---------------- object properties: (name, domain, range) ----------------
PROPS = [
 ('usesProtocol','Service','Protocol'), ('listensOn','Service','Port'), ('inRange','Port','PortRange'),
 ('commonlyAbusedFor','Port','AttackTechnique'), ('targetsService','AttackTechnique','Service'),
 ('enablesLateralMovement','Service','LateralMovement'), ('mitigatedBy','AttackTechnique','Control'),
 ('isPartOf','CloudAsset','CloudAsset'), ('runsOn','Container','CloudAsset'), ('assumesRole','ServiceAccount','IAMRole'),
 ('invokes','EventTrigger','Function'), ('routesTo','IngressController','Service'), ('governs','NetworkPolicy','Pod'),
 ('crossesBoundary','CrossTenantFlow','TenantBoundary'), ('exposes','CloudAsset','Service'),
 ('authenticatesWith','Service','AccessKey'), ('storesDataIn','Function','ObjectStorageBucket'),
 ('escalatesTo','AttackTechnique','AttackTechnique'), ('precedes','AttackTechnique','AttackTechnique'),
 ('protectedBy','Service','Control'),
]
# ---------------- services: name -> (application protocol, [ports], transport, abused-for) ----------------
SERVICES = {
 'FTP': ('FTP', [20, 21], 'TCP', ['BruteForce', 'DataExfiltration']),
 'SSH': ('SSH', [22], 'TCP', ['BruteForce', 'LateralMovement']),
 'Telnet': ('Telnet', [23], 'TCP', ['BruteForce', 'CommandAndControl']),
 'SMTP': ('SMTP', [25, 465, 587], 'TCP', ['DataExfiltration']),
 'DNS': ('DNS', [53], 'UDP', ['DenialOfService', 'DataExfiltration']),
 'DHCP': ('DHCP', [67, 68], 'UDP', ['DenialOfService']),
 'HTTP': ('HTTP', [80, 8000, 8080, 8888], 'TCP', ['WebAttack', 'DenialOfService']),
 'HTTPS': ('HTTPS', [443, 8443], 'TCP', ['WebAttack', 'DenialOfService', 'ProtocolExploit']),
 'Kerberos': ('Kerberos', [88], 'TCP', ['PrivilegeEscalation']),
 'POP3': ('POP3', [110, 995], 'TCP', ['BruteForce']),
 'IMAP': ('IMAP', [143, 993], 'TCP', ['BruteForce']),
 'NTP': ('NTP', [123], 'UDP', ['DenialOfService']),
 'MSRPC': ('MSRPC', [135], 'TCP', ['LateralMovement', 'ProtocolExploit']),
 'NetBIOS': ('NetBIOS', [137, 138, 139], 'UDP', ['LateralMovement']),
 'SNMP': ('SNMP', [161, 162], 'UDP', ['Reconnaissance']),
 'LDAP': ('LDAP', [389, 636], 'TCP', ['PrivilegeEscalation', 'Reconnaissance']),
 'SMB': ('SMB', [445], 'TCP', ['LateralMovement', 'ProtocolExploit']),
 'Syslog': ('Syslog', [514], 'UDP', []),
 'NFS': ('NFS', [2049], 'TCP', ['DataExposure']),
 'MSSQL': ('MSSQL', [1433], 'TCP', ['BruteForce', 'DataExfiltration']),
 'Oracle': ('Oracle', [1521], 'TCP', ['BruteForce', 'DataExfiltration']),
 'MySQL': ('MySQL', [3306], 'TCP', ['BruteForce', 'DataExfiltration']),
 'PostgreSQL': ('PostgreSQL', [5432], 'TCP', ['BruteForce', 'DataExfiltration']),
 'RDP': ('RDP', [3389], 'TCP', ['BruteForce', 'LateralMovement']),
 'VNC': ('VNC', [5900], 'TCP', ['BruteForce']),
 'Redis': ('Redis', [6379], 'TCP', ['DataExposure', 'Cryptojacking']),
 'MongoDB': ('MongoDB', [27017], 'TCP', ['DataExposure']),
 'Elasticsearch': ('Elasticsearch', [9200], 'TCP', ['DataExposure']),
 'IRC': ('IRC', [6667], 'TCP', ['CommandAndControl']),
 'DockerAPI': ('DockerAPI', [2375, 2376], 'TCP', ['ContainerEscape', 'Cryptojacking', 'ExposedAPIExploit']),
 'KubernetesAPI': ('KubernetesAPI', [6443], 'TCP', ['PrivilegeEscalation', 'ExposedAPIExploit']),
 'KubeletAPI': ('KubeletAPI', [10250], 'TCP', ['ContainerEscape', 'ExposedAPIExploit']),
 'Etcd': ('Etcd', [2379, 2380], 'TCP', ['DataExposure', 'ExposedAPIExploit']),
 'ContainerRegistryService': ('RegistryHTTP', [5000], 'TCP', ['SupplyChainCompromise']),
}
CONTROLS = {'BruteForce': 'RateLimiting', 'DenialOfService': 'TrafficFiltering', 'WebAttack': 'WebApplicationFirewall',
            'LateralMovement': 'NetworkSegmentation', 'ContainerEscape': 'RuntimeIsolation', 'Cryptojacking': 'ResourceQuota',
            'PrivilegeEscalation': 'LeastPrivilege', 'DataExposure': 'AccessControl', 'ExposedAPIExploit': 'NetworkPolicyEnforcement',
            'SupplyChainCompromise': 'ImageSigning', 'CommandAndControl': 'EgressFiltering', 'DataExfiltration': 'EgressFiltering',
            'Reconnaissance': 'PortScanDetection', 'ProtocolExploit': 'Patching'}
CHAINS = [('Reconnaissance', 'BruteForce'), ('BruteForce', 'LateralMovement'), ('LateralMovement', 'PrivilegeEscalation'),
          ('PrivilegeEscalation', 'DataExfiltration'), ('ContainerEscape', 'LateralMovement'), ('ExposedAPIExploit', 'ContainerEscape'),
          ('Cryptojacking', 'DenialOfService')]
CLOUD_PARTS = [('Container', 'Pod'), ('Pod', 'Deployment'), ('Sidecar', 'Pod'), ('Pod', 'DaemonSet'), ('Function', 'WorkflowOrchestrator')]

def build():
    triples = []          # (s, p, o) over local names; p in {'subClassOf','type', property names}
    cls = {c for c, _, _ in CLASSES}
    for c, parent, _ in CLASSES:
        if parent: triples.append((c, 'subClassOf', parent))
    inds = {}             # individual -> class
    def ind(name, klass):
        inds[name] = klass; triples.append((name, 'type', klass))
    for r in ('WellKnownRange', 'RegisteredRange', 'DynamicRange'): ind(r, 'PortRange')
    for n in range(0, 1024): ind(f'Port_{n}', 'Port'); triples.append((f'Port_{n}', 'inRange', 'WellKnownRange'))
    for t in ('TCP', 'UDP', 'ICMP'): ind(t, 'Protocol')
    for svc, (app, ports, tr, abused) in SERVICES.items():
        ind(f'{svc}Service', 'Service'); ind(f'{app}Protocol', 'Protocol')
        triples.append((f'{svc}Service', 'usesProtocol', f'{app}Protocol')); triples.append((f'{svc}Service', 'usesProtocol', tr))
        for p in ports:
            if f'Port_{p}' not in inds:
                ind(f'Port_{p}', 'Port'); triples.append((f'Port_{p}', 'inRange', 'RegisteredRange'))
            triples.append((f'{svc}Service', 'listensOn', f'Port_{p}'))
            for a in abused: triples.append((f'Port_{p}', 'commonlyAbusedFor', a))
        for a in abused:
            triples.append((a, 'targetsService', f'{svc}Service'))
            if a == 'LateralMovement': triples.append((f'{svc}Service', 'enablesLateralMovement', 'LateralMovement'))
    for a, c in CONTROLS.items(): ind(c, 'Control'); triples.append((a, 'mitigatedBy', c))
    for a, b in CHAINS: triples.append((a, 'precedes', b))
    triples.append(('ContainerEscape', 'escalatesTo', 'PrivilegeEscalation'))
    triples += [('KubernetesAPIServer', 'exposes', 'KubernetesAPIService'), ('Kubelet', 'exposes', 'KubeletAPIService')]
    triples += [('IngressController', 'routesTo', 'HTTPSService'), ('APIGateway', 'routesTo', 'HTTPSService'),
                ('NetworkPolicy', 'governs', 'Pod'), ('EventTrigger', 'invokes', 'Function'), ('ServiceAccount', 'assumesRole', 'IAMRole'),
                ('CrossTenantFlow', 'crossesBoundary', 'TenantBoundary'), ('Function', 'storesDataIn', 'ObjectStorageBucket'),
                ('Container', 'runsOn', 'Pod')]
    for a, b in CLOUD_PARTS: triples.append((a, 'isPartOf', b))
    return triples, inds

def write_ttl(path, triples, inds):
    props = {p for p, _, _ in PROPS}
    L = [f'@prefix cso: <{NS}> .', '@prefix owl: <http://www.w3.org/2002/07/owl#> .', '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
         '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .', '', f'<{NS[:-1]}> a owl:Ontology ; rdfs:label "CSO+ (thesis ontology)" .', '']
    for c, parent, theme in CLASSES:
        L.append(f'cso:{c} a owl:Class' + (f' ; rdfs:subClassOf cso:{parent}' if parent else '') + f' ; rdfs:comment "theme: {theme}" .')
    for p, d, r in PROPS: L.append(f'cso:{p} a owl:ObjectProperty ; rdfs:domain cso:{d} ; rdfs:range cso:{r} .')
    for i, k in inds.items(): L.append(f'cso:{i} a owl:NamedIndividual, cso:{k} .')
    for s, p, o in triples:
        if p in props: L.append(f'cso:{s} cso:{p} cso:{o} .')
    open(path, 'w').write('\n'.join(L) + '\n')

def graph(triples):
    nodes = sorted({x for s, p, o in triples for x in (s, o)})
    idx = {n: i for i, n in enumerate(nodes)}
    edges = {tuple(sorted((idx[s], idx[o]))) for s, p, o in triples if s != o}
    return nodes, idx, sorted(edges)

def node2vec(N, edges, dim=32, walk_len=10, walks=20, window=8, p=0.5, q=0.5, neg=5, epochs=10, seed=0, lr0=0.5):
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=bool)
    for a, b in edges: A[a, b] = A[b, a] = True
    nbrs = [np.flatnonzero(A[i]) for i in range(N)]
    walk_list = []
    for _ in range(walks):
        for start in rng.permutation(N):
            w = [start]
            if len(nbrs[start]) == 0: continue
            w.append(int(rng.choice(nbrs[start])))
            while len(w) < walk_len:
                prev, cur = w[-2], w[-1]; nb = nbrs[cur]
                if len(nb) == 0: break
                wt = np.where(nb == prev, 1 / p, np.where(A[prev, nb], 1.0, 1 / q))
                c = np.cumsum(wt); w.append(int(nb[np.searchsorted(c, rng.random() * c[-1])]))
            walk_list.append(w)
    centers, contexts = [], []
    for w in walk_list:
        L = len(w)
        for i in range(L):
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j != i: centers.append(w[i]); contexts.append(w[j])
    centers = np.array(centers); contexts = np.array(contexts)
    freq = np.bincount(centers, minlength=N).astype(np.float64) ** 0.75; freq /= freq.sum()
    Win = ((rng.random((N, dim)) - 0.5) / dim).astype(np.float32); Wout = np.zeros((N, dim), dtype=np.float32)
    B = 4096; total = epochs * (len(centers) // B + 1); step = 0
    for ep in range(epochs):
        perm = rng.permutation(len(centers))
        for s in range(0, len(perm), B):
            idx = perm[s:s + B]; c = centers[idx]; o = contexts[idx]
            lr = lr0 * max(0.05, 1 - step / total); step += 1
            ng = rng.choice(N, size=(len(idx), neg), p=freq)
            u = Win[c]                                   # [B,d]
            vpos = Wout[o]; vneg = Wout[ng]              # [B,d], [B,neg,d]
            lp = np.clip(np.sum(u * vpos, 1), -10, 10); ln = np.clip(np.einsum('bd,bnd->bn', u, vneg), -10, 10)
            sp = 1 / (1 + np.exp(-lp)); sn = 1 / (1 + np.exp(-ln))
            gpos = (sp - 1)[:, None]; gneg = sn[:, :, None]
            du = gpos * vpos + np.sum(gneg * vneg, 1)
            # average duplicated indices inside a batch (hub nodes appear many times)
            cc = np.maximum(1, np.bincount(c, minlength=N))[c][:, None]
            co = np.maximum(1, np.bincount(o, minlength=N))[o][:, None]
            cn = np.maximum(1, np.bincount(ng.ravel(), minlength=N))[ng.ravel()][:, None]
            np.add.at(Win, c, (-lr * du / cc).astype(np.float32))
            np.add.at(Wout, o, (-lr * (gpos * u) / co).astype(np.float32))
            np.add.at(Wout, ng.ravel(), (-lr * (gneg * u[:, None, :]).reshape(-1, dim) / cn).astype(np.float32))
        print(f'  skip-gram epoch {ep+1}/{epochs} done', flush=True)
    return Win, len(walk_list), len(centers)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out_dir', default='.'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    triples, inds = build(); write_ttl(f'{a.out_dir}/cso_plus.ttl', triples, inds)
    nodes, idx, edges = graph(triples)
    print(f'{len(nodes)} nodes, {len(edges)} undirected edges')
    emb, n_walks, n_pairs = node2vec(len(nodes), edges, epochs=a.epochs, seed=a.seed)
    emb = emb - emb.mean(axis=0, keepdims=True)          # remove the shared component
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    vec = np.zeros((65536, emb.shape[1]), dtype=np.float32)
    for n in range(65536):
        key = f'Port_{n}'
        if key in idx: vec[n] = emb[idx[key]]
        elif n <= 49151: vec[n] = emb[idx['RegisteredRange']]
        else: vec[n] = emb[idx['DynamicRange']]
    np.savez_compressed(f'{a.out_dir}/cso_plus_node2vec.npz', ports=np.arange(65536), vectors=vec)
    themes = defaultdict(list)
    for c, _, t in CLASSES: themes[t].append(c)
    json.dump(themes, open(f'{a.out_dir}/cso_plus_classes.json', 'w'), indent=1)
    stats = dict(classes=len(CLASSES), object_properties=len(PROPS), individuals=len(inds), triples=len(triples),
                 graph_nodes=len(nodes), graph_edges=len(edges), services=len(SERVICES), walks=n_walks, training_pairs=n_pairs,
                 dim=int(emb.shape[1]), walk_length=10, walks_per_node=20, window=8, p=0.5, q=0.5, negatives=5, epochs=a.epochs, seed=a.seed,
                 classes_by_theme={k: len(v) for k, v in themes.items()})
    json.dump(stats, open(f'{a.out_dir}/cso_plus_stats.json', 'w'), indent=1); print(json.dumps(stats, indent=1))
    # sanity check: nearest neighbours of some service ports (cosine)
    def near(port, k=6):
        v = vec[port]; ids = list(range(0, 1024)) + [p for p in (3306, 3389, 5432, 6379, 8080, 2375, 6443, 27017)]
        sims = sorted(((float(vec[i] @ v), i) for i in set(ids) if i != port), reverse=True)[:k]
        return [(i, round(s, 2)) for s, i in sims]
    for p_ in (22, 3389, 80, 3306, 2375): print('nearest to port', p_, near(p_))
    print('cos(22,21)=%.2f cos(22,80)=%.2f cos(22,4)=%.2f cos(80,443)=%.2f' % tuple(float(vec[a_] @ vec[b_]) for a_, b_ in [(22, 21), (22, 80), (22, 4), (80, 443)]))

if __name__ == '__main__':
    main()
