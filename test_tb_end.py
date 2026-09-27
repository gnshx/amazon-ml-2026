with open('tune_benchmark_50k.py') as f:
    lines = f.readlines()
for i, l in enumerate(lines[148:], 149):
    print(i, repr(l))

