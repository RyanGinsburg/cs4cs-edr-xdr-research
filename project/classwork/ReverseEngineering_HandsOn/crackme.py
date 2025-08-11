print("1.")
for i in range(5):
    print("*\n")

print()
print("2.")
for i in range(4):
    print("*")

print()
print("3.")
for i in range(1, 5):
    for j in range(i):
        print("*", end="")
    print()

print()
print("4.")
for i in range(1, 11):
    print(f'3 x {i} = {3 * i}')

print()
print("5.")
for i in range(5, 0, -1):
    for j in range(1, i+1):
        print(j, end="")
    print()