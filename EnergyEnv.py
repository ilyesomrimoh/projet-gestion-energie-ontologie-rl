import numpy as np
from ontology import create_ontology
from rdflib import Namespace

class EnergyEnv:
    def __init__(self):
        # Charger l’ontologie RDF
        self.g = create_ontology()
        self.ns = Namespace("http://example.org/energy#")
        # Extraire les informations des batteries et du moteur
        self.batteryA_charge = float(self.g.value(self.ns.BatteryA, self.ns.hasCharge))
        self.batteryB_charge = float(self.g.value(self.ns.BatteryB, self.ns.hasCharge))
        self.motor_demand = float(self.g.value(self.ns.Motor, self.ns.powerDemand))

        # Initialiser l’état du RL
        self.state = np.array([self.batteryA_charge, self.batteryB_charge])
        self.demand = self.motor_demand
        self.done = False

    def step(self, action):
        # action = 0 → utiliser batterie A
        # action = 1 → utiliser batterie B
        reward = 0
        consumption = self.motor_demand / 10  #  proportion de la demande moteur à chaque étape
        if action == 0:
            self.state[0] -= consumption
            self.state[0] = np.maximum(self.state[0], 0)
        else:
            self.state[1] -= consumption
            self.state[1] = np.maximum(self.state[1], 0)
        
        reward = 10 - abs(self.state[0] - self.state[1]) * 0.2

        if min(self.state) <= 0:
            self.done = True
            reward -= 10

        return self.state, reward

    def reset(self):
        # Remet les valeurs initiales depuis l’ontologie
        self.state = np.array([
            float(self.g.value(self.ns.BatteryA, self.ns.hasCharge)),
            float(self.g.value(self.ns.BatteryB, self.ns.hasCharge))
        ])
        self.done = False
        return self.state
