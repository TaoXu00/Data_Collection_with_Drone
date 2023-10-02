class Drone:
    def __init__(self, capacity, hovering_energy_per_unit, flying_energy_per_unit, comm_rate, speed, unit_time_uav_operation_cost):
        self.capacity = capacity
        self.hovering_energy_per_unit =hovering_energy_per_unit
        self.flying_energy_per_unit =flying_energy_per_unit
        self.comm_rate=comm_rate
        self.speed = speed  #m/s
        self.unit_time_uav_operation_cost = unit_time_uav_operation_cost  #dollar/s


