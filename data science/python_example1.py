# Write a code that receives a sentence and outputs the result according to the grammar
# Ex) python IS the best Language
# output) Python is the best language.

# 1. input variable definition
sentence = input('Input sentence: ')

# 2. all string convert lower
convert_lower = sentence.lower()
print(convert_lower)

# 3. Changed to the first capital letter
convert_capital = convert_lower[0].upper() + convert_lower[1:]
print(convert_capital)

# 4. add the string (.)
if convert_capital[-1] != ".":
    convert_capital += "."

# print
print('final sentence: {0}'.format(convert_capital))