class Parent:
    def __init__(self, name):
        self.name = name
        print('Parant __init__ start')
        print(self.name, '어서오고')
        print('Parant __init__ end')
    fool = 4
    
aaa = Parent('태영이')

class Child(Parent):
    def __init__(self, name):
        print('\nChild __init__ start')
        super().__init__(name)
        print('Child __init__ end')
        print('\nfool + genius :', super().fool + self.genius)
        print('fool + genius :', self.fool + self.genius)
    genius = 5
    
bbb = Child('예진이')
