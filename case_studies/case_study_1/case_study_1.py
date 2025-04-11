#initialize variables
a = 0
b = 0
res = 0

import random as rand

print('Simple Calculator')
print('Operations: +,-,*,/')

#get inputs
a = float(input("Enter first number that will be used as the first operand in the mathematical operation that you want to perform: "))  #first number
b = float(input('Enter second number that will be used as the second operand in the mathematical operation that you want to perform: '))  #second number
OP = input("Enter the desired mathematical operation that you would like to perform on the two numbers you just provided (+, -, *, /): ")   #what operation to do

#do the calculation
if OP=="+": #addition
    res=a+b
elif OP=="-": #subtraction
    res=a-b
elif OP=="*": #multiplication
    res=a*b
elif OP=="/": #division
    if b=="0":
        print('Cannot divide by zero') # error message 
    else:
        res=a/b
else:
    print('Invalid operation') # invalid operation message

print("Result:",res)
print('Thank you for using the calculator!')