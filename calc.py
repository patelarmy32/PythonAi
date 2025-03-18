def calculator():
    print("Basic Calc")
    print("Select operation")
    print("1. Addition")
    print("2. Subtract")
    print("3. Multiplication")
    print("4. Division")
    print("5.floor Division")
    print("6.modulus")
    choice = input("Enter choice (1/2/3/4/5/6): ")

    if choice in ('1', '2', '3', '4','5','6'):
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        if choice == '1':
            print(f"Result: {num1 + num2}")
        elif choice == '2':
            print(f"Result: {num1 - num2}")
        elif choice == '3':
            print(f"Result: {num1 * num2}")
        elif choice == '4':
            if num2 != 0:
                print(f"Result: {num1 / num2}")
            else:
                print("Error! Division by zero.")
        elif choice == '5':
            print(f"Result: {num1 // num2}")
        elif choice == '6':
            print(f"Result: {num1} % {num2}")

    else:
        print("Invalid input")

calculator()