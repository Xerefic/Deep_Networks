from imports import *

def complement(x):
    return torch.ones_like(x) - x

def AND_gate(x):
    t = torch.cat([y.unsqueeze(0) for y in x], dim=0)
    t = torch.prod(t, dim=0)
    return t

def OR_gate(x):
    if len(x) > 2:
        y = [OR_gate(x[0:2])]
        y.extend(x[2:])
        return OR_gate(y)
    if len(x) == 2:
        t = torch.cat([y.unsqueeze(0) for y in x], dim=0)
        t = torch.prod(complement(t), dim=0)
        t = complement(t)
        return t
    if len(x) == 1:
        return x[0]

def process_AND(x):
    if len(x) > 1:
        t = process_AND(x[1:])
        out = [x[0]*e for e in t]
        out.extend([complement(x[0])*e for e in t])
        return out
    elif len(x) == 1:
        return [x[0], complement(x[0])]

def process_OR(x):
    out = OR_gate(x)
    return out

def gate(x, exposure, gating=['ALL', None]):
    if gating[0] == 'OR':
        return OR_gate(x)
    if gating[0] == 'AND':
        return AND_gate(x)
    elif gating[0] == 'ALL':
        if len(x) > 2:
            t = x
            t = process_AND(x)
            t = list(itertools.compress(t, exposure))
            t = process_OR(t)
            return t
        else:
            return x[0]
    elif gating[0] == 'MIX':
        pure = [t * gating[1].to(t.device) for t in x]
        random = [t * complement(gating[1]).to(t.device) for t in x]
        pure = gate(pure, None, gating=['AND', None])
        random = gate(random, exposure, gating=['ALL', None])
        return gate([pure, random], None, gating=['OR', None])