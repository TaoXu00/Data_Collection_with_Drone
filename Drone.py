class Drone:
    def __init__(self, capacity, hovering_energy_per_unit, flying_energy_per_unit, comm_rate):
        self.capacity = capacity
        self.hovering_energy_per_unit =hovering_energy_per_unit
        self.flying_energy_per_unit =flying_energy_per_unit
        self.comm_rate=comm_rate
