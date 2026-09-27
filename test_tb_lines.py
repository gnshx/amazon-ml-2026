with open('tune_benchmark_50k.py') as f:
    lines = f.readlines()
for i in range(120, len(lines)):
    print(f'{i+1}: {repr(lines[i])}')

