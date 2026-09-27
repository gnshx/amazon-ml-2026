with open('src/run_robust_production_v4.py', 'r', encoding='utf-8') as f:
    text = f.read()
lines = text.splitlines()
print('Total lines:', len(lines))
for i in range(860, min(900, len(lines))):
    safe_l = lines[i].encode('ascii', 'backslashreplace').decode('ascii')
    print(f'{i+1}: {safe_l}')

