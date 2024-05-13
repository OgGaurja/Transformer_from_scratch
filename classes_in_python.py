class MyClass:
    #Constructor method
    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2


    #Instance method
    def my_method(self):
        return self.parameter1 + self.parameter2    

#creating an instance

my_instance = MyClass(10,40)

print(my_instance.parameter1)
print(my_instance.parameter2)