with open('src/run_robust_production_v4.py', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
print('Line 867 ords:', [ord(c) for c in lines[866][:50]])

