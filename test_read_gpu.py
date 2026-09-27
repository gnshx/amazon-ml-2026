with open('src/run_gpu_pipeline.py') as f:
    for i, line in enumerate(f):
        if i >= 60: break
        print(f'{i+1}: {line}', end='')

