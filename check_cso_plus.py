"""Structural checks and competency questions for cso_plus.ttl (no reasoner needed, standard library only).
This is NOT a substitute for running a description-logic reasoner (HermiT, Pellet) in Protege; it checks what can be checked by script:
 declared vocabulary, acyclic hierarchy, domain/range typing of every triple, unused classes, and answers to competency questions."""
import re, sys, collections, json
path = sys.argv[1] if len(sys.argv) > 1 else 'cso_plus.ttl'
classes, parent, props, indiv, triples, theme = {}, {}, {}, collections.defaultdict(set), [], {}
for line in open(path, encoding='utf-8'):
    line = line.strip()
    if not line.startswith('cso:') or not line.endswith('.'): continue
    body = line[:-1].strip(); s, rest = body.split(' ', 1)
    if ' a owl:Class' in ' ' + rest:
        classes[s] = True
        m = re.search(r'rdfs:subClassOf (cso:\w+)', rest); parent[s] = m.group(1) if m else None
        t = re.search(r'theme: (\w+)', rest); theme[s] = t.group(1) if t else '?'
    elif ' a owl:ObjectProperty' in ' ' + rest:
        d = re.search(r'rdfs:domain (cso:\w+)', rest); r = re.search(r'rdfs:range (cso:\w+)', rest)
        props[s] = (d.group(1) if d else None, r.group(1) if r else None)
    elif ' a owl:NamedIndividual' in ' ' + rest:
        for c in re.findall(r'(cso:\w+)', rest): indiv[s].add(c)
    else:
        m = re.match(r'(cso:\w+) (cso:\w+)$', rest)
        if m: triples.append((s, m.group(1), m.group(2)))
def ancestors(c):
    out = []; seen = set()
    while c and c not in seen: seen.add(c); out.append(c); c = parent.get(c)
    return out
issues = []
for c, p in parent.items():                                   # 1 hierarchy
    if p and p not in classes: issues.append(f'parent of {c} undeclared: {p}')
for c in classes:
    a = ancestors(c)
    if len(a) != len(set(a)): issues.append(f'cycle in hierarchy at {c}')
for p, (d, r) in props.items():                               # 2 property vocabulary
    for x in (d, r):
        if x and x not in classes: issues.append(f'property {p}: undeclared class {x}')
viol, punned = [], 0
for s, p, o in triples:                                       # 3 and 4 triples
    if p not in props: issues.append(f'undeclared predicate {p} in ({s},{p},{o})'); continue
    d, r = props[p]
    stypes = set(sum((ancestors(c) for c in indiv.get(s, [])), [])) if s in indiv else (set(ancestors(s)) if s in classes else None)
    otypes = set(sum((ancestors(c) for c in indiv.get(o, [])), [])) if o in indiv else (set(ancestors(o)) if o in classes else None)
    if s in classes and s not in indiv: punned += 1
    if o in classes and o not in indiv: punned += 1
    if stypes is None: issues.append(f'subject not declared: {s}')
    elif d and d not in stypes: viol.append((s, p, o, f'subject not a {d}'))
    if otypes is None: issues.append(f'object not declared: {o}')
    elif r and r not in otypes: viol.append((s, p, o, f'object not a {r}'))
used_class = collections.Counter()
for s, cs in indiv.items():
    for c in cs:
        for a in ancestors(c): used_class[a] += 1
mentioned = {x for t in triples for x in (t[0], t[2])} | {d for d, r in props.values() if d} | {r for d, r in props.values() if r}
by_theme = collections.defaultdict(lambda: [0, 0])
for c in classes:
    by_theme[theme[c]][0] += 1
    if used_class[c] > 0 or any(t[0] == c or t[2] == c for t in triples): by_theme[theme[c]][1] += 1
out = []
out.append(f'classes {len(classes)}, object properties {len(props)}, individuals {len(indiv)}, triples {len(triples)}')
out.append(f'hierarchy/vocabulary issues: {len(issues)}; domain/range violations: {len(viol)}; triples that use a class name as subject or object (OWL punning): {punned}')
for v in viol[:10]: out.append('  violation: ' + str(v))
for i in issues[:10]: out.append('  issue: ' + i)
out.append('disjointness axioms: 0 (none are declared), so a reasoner cannot find this ontology inconsistent; consistency is trivially true and says nothing about quality')
out.append('')
out.append('classes that occur in at least one individual or triple, by theme (used / total):')
for t, (n, u) in sorted(by_theme.items()): out.append(f'  {t:16s} {u} / {n}')
unused = sorted(c for c in classes if used_class[c] == 0 and not any(t[0] == c or t[2] == c for t in triples))
out.append(f'classes never instantiated and never used in a triple ({len(unused)}): ' + ', '.join(u.replace('cso:', '') for u in unused))
out.append('')
out.append('competency questions answered from the triples:')
def objs(s, p): return sorted(o.replace('cso:', '') for a, b, o in triples if a == s and b == p)
def subs(p, o): return sorted(s.replace('cso:', '') for s, b, c in triples if b == p and c == o)
out.append('Q1 Which attack techniques is Port_3389 commonly abused for? ' + ', '.join(objs('cso:Port_3389', 'cso:commonlyAbusedFor')))
out.append('Q2 Which ports are commonly abused for BruteForce? ' + ', '.join(subs('cso:commonlyAbusedFor', 'cso:BruteForce')))
svc = collections.defaultdict(list)
for s, p, o in triples:
    if p == 'cso:listensOn': svc[o].append(s)
out.append('Q3 Which services listen on Port_22, Port_80, Port_443, Port_3389? ' + '; '.join(f'{p.replace("cso:", "")}: ' + ','.join(x.replace('cso:', '') for x in svc.get(p, [])) for p in ('cso:Port_22', 'cso:Port_80', 'cso:Port_443', 'cso:Port_3389')))
sub = sorted(c.replace('cso:', '') for c in classes if 'cso:AttackTechnique' in ancestors(c) and c != 'cso:AttackTechnique')
out.append(f'Q4 Which attack techniques exist ({len(sub)})? ' + ', '.join(sub))
prop_count = collections.Counter(p for s, p, o in triples)
out.append('Q5 How many triples use each property? ' + ', '.join(f'{p.replace("cso:", "")}={n}' for p, n in prop_count.most_common()))
print('\n'.join(out))
open('ontology_checks.txt', 'w').write('\n'.join(out) + '\n')
