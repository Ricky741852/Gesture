with open('trainData/2/2_2024-02-21-03-51-04_RICKY_TR.txt', 'r') as file:
    for i, line in enumerate(file):
        try:
            list(map(int, line.split(",")))
        except ValueError:
            print(f"Cannot convert line {i+1} to integer: {line.strip()}")