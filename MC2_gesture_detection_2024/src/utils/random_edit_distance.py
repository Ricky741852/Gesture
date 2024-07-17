import random

# 生成10個隨機組合
random_combinations = [random.choices(range(1, 4), k=4) for _ in range(10)]

for combo in random_combinations:
    print(combo)