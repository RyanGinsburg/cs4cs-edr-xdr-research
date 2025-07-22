def calculate(num1, operation, num2):
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        return num1 / num2
    else:
        return None
    
while True:
    num1 = int(input("Enter the first number: "))
    operation = input("Enter an operation (+, -, *, /): ")
    num2 = int(input("Enter the second number: "))

    result = calculate(num1, operation, num2)
    if result is None:
        print("Invalid operation. Please try again.")
        continue

    print(f"Result: {result}")
    continue_calculation = input("Do you want to calculate again? (yes/no): ")
    if continue_calculation.lower() != 'yes':
        break
