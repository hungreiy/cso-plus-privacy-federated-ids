"""Competency-question evaluation of CSO+ (Section 4.x): every question is answered by a SPARQL query on cso_plus.ttl.
  python ontology_cq.py ../ontology/cso_plus.ttl   -> prints the answers and writes ontology_cq_results.json"""
import sys, json
from rdflib import Graph
g = Graph(); g.parse(sys.argv[1], format='turtle')
P = 'PREFIX cso: <http://example.org/csoplus#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX owl: <http://www.w3.org/2002/07/owl#> '
CQ = [
 ('CQ1', 'Which attack techniques are commonly carried out over port 3389?',
  'SELECT ?t WHERE { cso:Port_3389 cso:commonlyAbusedFor ?t }'),
 ('CQ2', 'Which service listens on port 3389, and over which protocols?',
  'SELECT ?s ?pr WHERE { ?s cso:listensOn cso:Port_3389 . OPTIONAL { ?s cso:usesProtocol ?pr } }'),
 ('CQ3', 'Which ports are commonly abused for brute-force attacks?',
  'SELECT ?p WHERE { ?p cso:commonlyAbusedFor cso:BruteForce }'),
 ('CQ4', 'Which services enable lateral movement?',
  'SELECT DISTINCT ?s WHERE { { ?s cso:enablesLateralMovement ?x } UNION { ?s cso:listensOn ?p . ?p cso:commonlyAbusedFor cso:LateralMovement } }'),
 ('CQ5', 'Which controls mitigate brute-force attacks?',
  'SELECT ?c WHERE { cso:BruteForce cso:mitigatedBy ?c }'),
 ('CQ6', 'Which attack techniques can follow a scan in an attack sequence?',
  'SELECT ?t2 WHERE { ?t1 cso:precedes ?t2 . FILTER(CONTAINS(LCASE(STR(?t1)), "scan") || CONTAINS(LCASE(STR(?t1)), "recon")) }'),
 ('CQ7', 'Which container-platform services are described, and on which ports do they listen?',
  'SELECT ?s ?p WHERE { VALUES ?s { cso:KubernetesAPIService cso:KubeletAPIService cso:EtcdService cso:DockerAPIService cso:ContainerRegistryService } OPTIONAL { ?s cso:listensOn ?p } }'),
 ('CQ8', 'Which attack techniques target the Kubernetes API?',
  'SELECT DISTINCT ?t WHERE { { ?t cso:targetsService cso:KubernetesAPIService } UNION { cso:Port_6443 cso:commonlyAbusedFor ?t } }'),
 ('CQ9', 'Which services authenticate with access keys, and which controls protect a service?',
  'SELECT ?x ?y WHERE { { ?x cso:authenticatesWith ?y } UNION { ?x cso:protectedBy ?y } }'),
 ('CQ10', 'On which ports does the container network of a Kubernetes cluster (OVN-Kubernetes) listen?',
  'SELECT ?s ?p WHERE { ?s cso:listensOn ?p . FILTER(CONTAINS(LCASE(STR(?s)), "ovn")) }'),
]
short = lambda v: str(v).split('#')[-1]
out = []
for cid, q, sp in CQ:
    try:
        rows = [[short(v) for v in r if v is not None] for r in g.query(P + sp)]
        ans = sorted({' / '.join(r) for r in rows if r}); ok = len(ans) > 0; err = None
    except Exception as e:
        ans, ok, err = [], False, str(e)
    out.append(dict(id=cid, question=q, sparql=sp, answered=ok, answers=ans, error=err))
    print(f"{cid} [{'answered' if ok else 'NOT answered'}] {q}\n     -> {ans[:12]}{' ...' if len(ans) > 12 else ''}{'  ERROR: ' + err if err else ''}")
use = {short(p): int(n) for p, n in g.query(P + 'SELECT ?p (COUNT(*) AS ?n) WHERE { ?p a owl:ObjectProperty . OPTIONAL { ?s ?p ?o } } GROUP BY ?p')}
unused = sorted(k for k, v in use.items() if v <= 1)   # COUNT(*) is 1 for a property with no assertions (the OPTIONAL row)
print('\nobject properties:', len(use), '| with no assertions:', unused)
print('triples:', len(g), '| classes:', len(set(g.subjects(predicate=None, object=None))) and len(list(g.query(P + 'SELECT ?c WHERE { ?c a owl:Class }'))))
json.dump(dict(questions=out, property_use=use, properties_without_assertions=unused, triples=len(g)), open('ontology_cq_results.json', 'w'), indent=2)
print(f"\n{sum(o['answered'] for o in out)} of {len(out)} competency questions answered; results in ontology_cq_results.json")
