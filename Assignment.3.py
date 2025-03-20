age = int(input("Enter the patient's age: "))

if age >= 15:
    if age < 18:
        weight = float(input("Enter the patient's weight (kg): "))
        if weight >= 55:
            print("Medicine can be given.")
        else:
            print("Medicine cannot be given.")
    else:
        print("Medicine can be given.")
else:
    print("Medicine cannot be given.!")
