age = int(input("Enter your age: "))
nationality = input("Are you a citizen? (Yes/No): ").strip().capitalize()  # Handles case sensitivity

if age >= 18 and nationality == "Yes":
    print(f"You are {age} years old. You are valid for voting.")
elif age >= 18 and nationality == "No":
    print(f"You are {age} years old, but since you have no nationality, you are not valid for voting.")
else:
    print(f"You are {age} years old and considered a minor. You are not valid for voting.")