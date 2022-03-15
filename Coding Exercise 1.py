# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.9
#     language: python
#     name: python3
# ---

# +
for _ in range(1, 4):
    print("This is Spartaaaaa!")


# +
def add_two_numbers(a,b):
    return a+b

add_two_numbers(2,5)


# -

class Phone:
    
    def make_call(self):
        print("Making a phone call.")
    
    def play_game(self):
        print("Playing game.")


p1 = Phone()

p1.make_call()

p1.play_game()


# +
# new class

class Phone:
    
    def add_colour(self,colour):
        self.colour = colour
    
    def add_cost(self,cost):
        self.cost = cost
    
    def show_colour(self):
        return self.colour
    
    def show_cost(self):
        return self.cost
    
    def make_call(self):
        print("Making a phone call.")
    
    def play_game(self):
        print("Playing game.")


# -

p2 = Phone()

p2.add_colour("blue")
p2.add_cost(200)

p2.show_colour()

p2.show_cost()


class iPhone(Phone):
    
    def cure_cancer(self):
        print("I can cure cancer.")


iP1 = iPhone()

iP1.add_colour("green")

iP1.add_cost(899)

iP1.cure_cancer()


