import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
os.chdir('C:/Users/U77225D/Desktop/proyectos/hf/builds/trace2grammar/experiments/elda')
pids = [json.loads(l)['pid'] for l in open('annotations/descriptions.jsonl', encoding='utf-8')]
print('n windows:', len(pids))
print(Counter(p.split(':')[0] for p in pids))
convs = sorted(set(p.rsplit(':',1)[0] for p in pids))
print('convs:', len(convs), '| sample:', convs[:8])
ts = [json.loads(l)['pid'] for l in open('annotations/typed_scores.jsonl', encoding='utf-8')]
print('typed windows:', len(ts), '| pids match:', set(ts)==set(pids))
for c in ['windows.jsonl','../windows.jsonl','../../windows.jsonl']:
    print(c, os.path.exists(c))
