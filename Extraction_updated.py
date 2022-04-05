import numpy as np
from stl import mesh
import math
import pandas as pd
from mpi4py import MPI
from shapely.geometry import Point, Polygon

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
file = pd.read_csv("output_street.csv")
data = np.array(file['shapeid'].unique())
print(data)
grid_list = (np.array_split(data,size))[rank]

print(file.X_Coordinate)

nodalmatrix = np.zeros((2,2))  # A matrix that stores the coordinates of the two points
your_mesh = mesh.Mesh.from_file('Region_of_Interest.stl')

cogs = np.zeros((your_mesh.data.size, 5)) #This section converts every triangle in a point (Center of gravity)
for i in range(0, your_mesh.data.size): #i refers to the rows of the array cog
    for j in range(0, 4):  # i refers to the columns of the array cog
        if j == 0:  #j=1 always refers to the calculation of the x cog coordinate for the triangle i
           cogs[i,j]=(your_mesh.x[i,0] + your_mesh.x[i,1] + your_mesh.x[i,2])/3
        elif j == 1:
           cogs[i,j] = (your_mesh.y[i,0] + your_mesh.y[i,1] + your_mesh.y[i,2])/3
        elif j == 2:
           cogs[i,j] = (your_mesh.z[i,0] + your_mesh.z[i,1] + your_mesh.z[i,2])/3
        elif j == 3:
           cogs[i,j] = i #This column stores the triangle ID
#print(cogs)
k = 0
streettriangle =[]
for i in range(2125):

    if file.shapeid[i] == grid_list[k] and file.shapeid[i+1] == grid_list[k]:
        '''for i in range(2):  # A for loop for the number of nodes
            print("Enter the coordinates of node ", i+1)
            coordinate = []
            for j in range(2):  # A for loop for the coordinates of each node
                coordinate.append(int(input()))
            nodalmatrix.append(coordinate)'''
        nodalmatrix[0,0] = file.X_Coordinate[i]
        nodalmatrix[0,1] = file.Y_Coordinate[i]
        nodalmatrix[1,0] = file.X_Coordinate[i+1]
        nodalmatrix[1,1] = file.Y_Coordinate[i+1]

        delta_x = nodalmatrix[0,0] - nodalmatrix[1,0]
        delta_y = nodalmatrix[0,1] - nodalmatrix[1,1]
        theta = math.atan(delta_y/delta_x)
        x1 = nodalmatrix[0,0] + (5 * math.sin(theta))
        y1 = nodalmatrix[0,1] - (5 * math.cos(theta))
        x2 = nodalmatrix[0,0] - (5 * math.sin(theta))
        y2 = nodalmatrix[0,1] + (5 * math.cos(theta))
        x3 = nodalmatrix[1,0] + (5 * math.sin(theta))
        y3 = nodalmatrix[1,1] - (5 * math.cos(theta))
        x4 = nodalmatrix[1,0] - (5 * math.sin(theta))
        y4 = nodalmatrix[1,1] + (5 * math.cos(theta))
        coords = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = Polygon(coords)
        for j in range(0, your_mesh.data.size):
            pt = Point(cogs[j,0],cogs[j,1])
            if pt.within(poly):
                street =np.array([j,your_mesh.x[j,0],your_mesh.y[j,0],your_mesh.z[j,0],your_mesh.x[j,1],your_mesh.y[j,1],your_mesh.z[j,1],your_mesh.x[j,2],your_mesh.y[j,2],your_mesh.z[j,2]])
                streettriangle.append(street)

    elif file.shapeid[i] < grid_list[k]:
        continue

    else:
        rows = len(streettriangle)
        generate = np.zeros(rows, dtype=mesh.Mesh.dtype)
        for l in range(rows):
            generate['vectors'][l] = np.array([[streettriangle[l][1], streettriangle[l][2], streettriangle[l][3]],
                                               [streettriangle[l][4], streettriangle[l][5], streettriangle[l][6]],
                                               [streettriangle[l][7], streettriangle[l][8], streettriangle[l][9]]])
        generate_street = mesh.Mesh(generate)
        print("Street ",k)
        filename = "Street" + "_" + str(rank) + "_" + str(k) + ".stl"
        generate_street.save(filename)
        streettriangle = []
        if k<((grid_list.size) - 1):
            k=k+1
        else:
            break

