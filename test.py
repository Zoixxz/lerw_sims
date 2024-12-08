total = set()

for x in range(-8, 8 + 1):
    for y in range(-8, 8 + 1):
        if abs(x) == 8 or abs(y) == 8:
            total.add((x, y))

print(len(total))