# Write code to generate 6 digit lotto numbers
# 6 digit number should not be duplicated

# Moudule import 
import random

# 6 digit lotto numbers Generate
lotto_numbers = []

while True:
    numbers = random.randint(1,45)
    if numbers not in lotto_numbers:
        lotto_numbers.append(numbers)
    if len(lotto_numbers) >= 6:
        lotto_numbers.sort()
        break
print(lotto_numbers)