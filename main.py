import pygame
import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.flag = None  # Green, Yellow, Red
        self.cluster = -1

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

def add_near_point(point, radius=15):
    new_points = []
    count = np.random.randint(2, 5)
    for _ in range(count):
        angle = np.random.rand() * 2 * math.pi
        distance = np.random.rand() * radius
        new_x = point.x + math.cos(angle) * distance
        new_y = point.y + math.sin(angle) * distance

        new_x = max(0, min(600, new_x))
        new_y = max(0, min(400, new_y))

        new_points.append(Point(new_x, new_y))
    return new_points

def get_array(points):
    values = np.zeros((len(points), 2))
    for i in range(len(points)):
        values[i, 0] = points[i].x
        values[i, 1] = points[i].y
    return values

def draw_points(screen, points, r):
    for point in points:
        color = "black"
        if point.flag == "green":
            color = "green"
        elif point.flag == "yellow":
            color = "yellow"
        elif point.flag == "red":
            color = "red"

        pygame.draw.circle(screen, color, (int(point.x), int(point.y)), r)

def dbscan(points, eps, min_pts):
    cluster_id = 0
    for point in points:
        if point.flag is not None:
            continue

        neighbors = region_query(points, point, eps)

        if len(neighbors) < min_pts:
            # Если хотя бы один сосед граничная точка и соседей недостаточно - точка шумовая
            if any(neighbor.flag in ["yellow", "red"] for neighbor in neighbors) or len(neighbors) == 0 :
                point.flag = "red"
            else:
                point.flag = "yellow"
            continue


        point.flag = "green"
        point.cluster = cluster_id

        seeds = neighbors[:]
        for neighbor in seeds:
            if neighbor.flag == "yellow":
                neighbor.flag = "green"
                neighbor.cluster = cluster_id
            if neighbor.flag is not None:
                continue
            neighbor.flag = "green"
            neighbor.cluster = cluster_id

            new_neighbors = region_query(points, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                seeds.extend(new_neighbors)

        cluster_id += 1


def region_query(points, center, eps):
    return [point for point in points if center.distance(point) <= eps]

def main():
    colors = ["blue", "green", "cyan", "yellow", "purple", "red"]
    r = 3
    points = []
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill(color="white")
    pygame.display.update()
    is_pressed = False
    eps = 30
    min_pts = 10

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dbscan1 = DBSCAN(eps=eps, min_samples=min_pts)
                    dbscan1.fit(get_array(points))
                    labels = dbscan1.labels_
                    plt.scatter(get_array(points)[:, 0], get_array(points)[:, 1], c=labels)
                    plt.show()
                if event.key == pygame.K_RETURN:
                    dbscan(points, eps, min_pts)
                    screen.fill("white")
                    draw_points(screen, points, r)
                    pygame.display.flip()
                if event.key == pygame.K_ESCAPE:
                    screen.fill("white")

                    for point in points:
                        if point.cluster >= 0:  # Ядро
                            color = colors[point.cluster % len(colors)]
                            pygame.draw.circle(
                                screen, color, (int(point.x), int(point.y)), r
                            )
                        else:  # Шум
                            pygame.draw.circle(
                                screen, "gray", (int(point.x), int(point.y)), r
                            )

                    pygame.display.flip()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_pressed = False
            if event.type == pygame.MOUSEMOTION and is_pressed:
                pos = event.pos
                p = Point(*pos)
                new_points = add_near_point(p)
                if len(points) == 0:
                    points.append(p)
                    pygame.draw.circle(screen, "black", pos, r)
                elif p.distance(points[-1]) >= 30:
                    points.append(p)
                    pygame.draw.circle(screen, "black", pos, r)
                for i in range(len(new_points)):
                    pygame.draw.circle(screen, "black", (int(new_points[i].x), int(new_points[i].y)), r)
                    points.append(new_points[i])
            pygame.display.flip()



if __name__ == "__main__":
    main()
