import json
import math
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

class tsp_solver:
    def __init__(self, nodes, coordinates):
        self.nodes=nodes
        self.coordinates=coordinates
    def calculte_distance(self):
        dist={}
        for n1, n2 in combinations(self.nodes, 2):
            c1 = self.coordinates[n1]
            c2 = self.coordinates[n2]
            diff = (c1[0] - c2[0], c1[1] - c2[1])
            distance= math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
            dist[(n1,n2)]=distance
        return dist

    def define_model(self, dist):
        m = gp.Model()
        # Variables: is city 'i' adjacent to city 'j' on the tour?
        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')

        # Symmetric direction: Copy the object
        for i, j in vars.keys():
            vars[j, i] = vars[i, j]  # edge in opposite direction

        # Constraints: two edges incident to each city
        cons = m.addConstrs(vars.sum(c, '*') == 2 for c in self.nodes)
        m._vars=vars
        return m, vars
    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(self,model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                    if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = self.subtour(selected)
            if len(tour) < len(self.nodes):
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                         <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(self,edges):
        #print("self.nodes %s" %(self.nodes))
        unvisited = self.nodes[:]
        cycle = self.nodes[:]  # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle  # New shortest subtour
        return cycle

    def solve(self):
        #self.nodes=attr
        dict=self.calculte_distance()
        m, vars=self.define_model(dict)
        #m._vars=vars
        m.Params.lazyConstraints = 1
        m.optimize(lambda model, where: self.subtourelim(model,where))
        # retrieve the solution
        vals = m.getAttr('x',vars)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = self.subtour(selected)
        assert len(tour) == len(self.nodes)
        optimal_dis=m.getObjective().getValue()
        #print("The tour is %s, total distance %f" % (tour, optimal_dis) )
        return tour, optimal_dis