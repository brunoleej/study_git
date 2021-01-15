# StarCraft Marine is designed as a class
# health : 40, attack_power : 5, attack
# Write the code that Marine 1 attacks Marine 2 by creating 2 Marine objects with Marine class.

# Class Generate
class Marine:
    def __init__(self,health=40,attack_power=5):
        self.health = 40
        self.attack_power = 5

    def attack(self,unit):
        unit.health -= self.attack_power
        if unit.health <= 0:
            unit.health = 0
            print('Die')

marine1 = Marine()
marine2 = Marine()
print(marine1.attack(marine2))
print('marine1.health: {0}'.format(marine1.health))
print('marine2.health: {0}'.format(marine2.health))


