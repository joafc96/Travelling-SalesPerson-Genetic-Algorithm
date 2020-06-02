import math
import random
import numpy as np
import matplotlib.pyplot as plt


class City:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Route:

    def __init__(self, cities):
        self.route = random.sample(cities, len(cities))
        self.fitness = None
        self.distance = None

    def __str__(self):
        return f'Route is {self.route}, Fitness is {self.fitness} and Distance is {self.distance}'


total_city = None
population = None
generations = None


def tsp_ga():
    shortest_distance = math.inf
    best_in_gen = []
    best_ever_route = None

    city_list = init_city_coord(total_city)  # initialising the total no of cities

    routes = init_routes(population, city_list)  # initialising the first set of routes

    for generation in range(generations):
        print(f'Generation is {generation}')

        routes = fitness(routes)  # calculate the fitness of the route
        routes = normalize_fitness(routes)  # normalize their fitness score so they adds up to 1

        for route in routes:
            if route.distance < shortest_distance:
                best_in_gen.append((generation, route.distance))
                shortest_distance = route.distance
                best_ever_route = route

        routes = selection(routes)  # select the best parents from the existing gen
        routes = crossover(routes, city_list)  # breed between the selected parents
        routes = mutation(routes)  # a very small percent of mutation is provided to the new born

    plot_graph(best_in_gen, best_ever_route)


def fitness(routes):
    return [calc_fitness(route) for route in routes]


def normalize_fitness(routes):
    max_fitness = sum([c.fitness for c in routes])

    for idx in range(len(routes)):
        routes[idx].fitness /= max_fitness
    return routes


def selection(routes):
    return [select_one(routes) for _ in range(int(0.2 * len(routes)))]


def crossover(routes, city_list):
    offsprings = []

    for _ in range(population - len(routes)):
        parent1 = random.choice(routes)
        parent2 = random.choice(routes)

        start = random.randint(0, len(city_list) - 1)
        end = random.randint(start + 1, len(city_list))

        child = Route(cities=city_list)
        child.route = []
        child.route.extend(parent1.route[start:end])

        for i in range(len(parent2.route)):
            child.route.append(parent2.route[i]) if parent2.route[i] not in child.route else child.route
        offsprings.append(child)
    routes.extend(offsprings)
    return routes


def mutation(routes):
    for route_obj in routes:
        for swapped, param in enumerate(route_obj.route):
            if random.uniform(0.0, 1.0) <= 0.01:
                swap_with = math.floor(random.random() * len(route_obj.route))
                route_obj.route[swapped], route_obj.route[swap_with] = route_obj.route[swap_with], route_obj.route[swapped]
    return routes


def init_city_coord(total):
    return [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(total)]


def init_routes(pop, cities):
    return [Route(cities) for _ in range(pop)]


def calc_fitness(route_obj):
    if route_obj.distance is None:
        path_distance = 0
        for i in range(len(route_obj.route)):
            from_city = route_obj.route[i]
            to_city = None
            if i + 1 < len(route_obj.route):
                to_city = route_obj.route[i + 1]
            else:
                to_city = route_obj.route[0]
            path_distance += from_city.distance(city=to_city)
        route_obj.distance = path_distance
        if route_obj.fitness is None:
            route_obj.fitness = 1 / float(route_obj.distance)
    return route_obj


def select_one(routes):
    max_fitness = sum([c.fitness for c in routes])
    selection_prob = [c.fitness / max_fitness for c in routes]
    return routes[np.random.choice(len(routes), p=selection_prob)]


def plot_graph(best_in_gen, best_ever_route):
    # scatter plot of the coordinates with the shortest distance
    x_list, y_list = [city.x for city in best_ever_route.route], [city.y for city in best_ever_route.route]
    plt.figure()
    plt.plot(x_list, y_list, 'r', zorder=1, lw=3)
    plt.scatter(x_list, y_list, s=120, zorder=2)
    plt.title('Shortest path')
    plt.show()

    # plot to show the convergence
    plt.figure()
    plt.plot(*zip(*best_in_gen))
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == '__main__':
    total_city = 10
    population = 300
    generations = 250
    tsp_ga()
