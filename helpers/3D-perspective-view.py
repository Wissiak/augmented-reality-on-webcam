import matplotlib.pyplot as plt
import numpy as np

points = np.array(
        [
            [0, 0, 0],
            [0.1, 0, 0],
            [0.1, 0, 0.1],
            [0, 0, 0.1],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0.1, 0.1, 0.1],
            [0, 0.1, 0.1],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1]
        ])

edges = [
         [4, 5], [5, 6], [6, 7], [7, 4],  # Lines of back plane
         [0, 4], [1, 5], [2, 6], [3, 7], # Lines connecting front with back-plane
         [0, 1], [1, 2], [2, 3], [3, 0],  # Lines of front plane
         [0, 8], [0, 9], [0, 10],  # Lines indicating the coordinate frame
        ]

edgecolors = [
               '0','0','0','0',
               '0','0','0','0',
               'k','k','k','k',
               'r','g','b'
             ]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(edges)):
    edge = edges[i]
    start = points[edge[0]]
    end = points[edge[1]]

    ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=edgecolors[i])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()