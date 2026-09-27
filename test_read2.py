with open('src/run_robust_production_v4.py') as f:
    lines = f.readlines()
for i, l in enumerate(lines[865:945], 866):
    print(f'{i}: {l}', end='')

