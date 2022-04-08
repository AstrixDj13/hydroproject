import numpy as np
import csv
from stl import mesh
import math
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
file = pd.read_csv("output_street.csv")     #Reading the csv file containing street nodes
data = np.array(file['shapeid'].unique())   #storing the unique shape-ids in an array
#print(data)
grid_list = (np.array_split(data,size))[rank]

#print(file.X_Coordinate)

nodalmatrix = np.zeros((2,2))  # A matrix that stores the coordinates of the two nodes of a street
your_mesh = mesh.Mesh.from_file('Region_of_Interest.stl')   #STL file from which streets are to be extracted

cogs = np.zeros((your_mesh.data.size, 5)) #This section converts every triangle to a point (Center of gravity)
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
for i in range(2125):     #loop which runs through all the the points in the csv file

    if file.shapeid[i] == grid_list[k] and file.shapeid[i+1] == grid_list[k]:
        '''for i in range(2):  # A for loop for the number of nodes
            print("Enter the coordinates of node ", i+1)
            coordinate = []
            for j in range(2):  # A for loop for the coordinates of each node
                coordinate.append(int(input()))
            nodalmatrix.append(coordinate)'''
        nodalmatrix[0,0] = file.X_Coordinate[i]     #Stores the X and Y cooedinates of the
        nodalmatrix[0,1] = file.Y_Coordinate[i]     #two nodes
        nodalmatrix[1,0] = file.X_Coordinate[i+1]   #in the nodalmatrix
        nodalmatrix[1,1] = file.Y_Coordinate[i+1]   #numpy array

        delta_x = nodalmatrix[0,0] - nodalmatrix[1,0]
        delta_y = nodalmatrix[0,1] - nodalmatrix[1,1]
        theta = math.atan(delta_y/delta_x)
        x1 = nodalmatrix[0,0] + (5 * math.sin(theta))   #Calculates the 4 corner coordinates of
        y1 = nodalmatrix[0,1] - (5 * math.cos(theta))   #rectangle formed by taking a buffer
        x2 = nodalmatrix[0,0] - (5 * math.sin(theta))   #of 5 on both sides 
        y2 = nodalmatrix[0,1] + (5 * math.cos(theta))
        x3 = nodalmatrix[1,0] + (5 * math.sin(theta))
        y3 = nodalmatrix[1,1] - (5 * math.cos(theta))
        x4 = nodalmatrix[1,0] - (5 * math.sin(theta))
        y4 = nodalmatrix[1,1] + (5 * math.cos(theta))
        '''print("The bounding coordinates are:  ")
        print(x1, ",", y1)
        print(x2, ",", y2)
        print(x3, ",", y3)
        print(x4, ",", y4)'''
        # Determination of whether the triangle centroid is inside or outside
        trianglecoordinate = np.zeros((1, 2))

        '''print("ID of the triangles inside this street:  ")
        counter=0'''
        
        for j in range(0, your_mesh.data.size):     #loop which runs through all the triangles in the mesh
            trianglecoordinate[0,0] = cogs[j,0]     #stores the coordinates of the centroid
            trianglecoordinate[0,1] = cogs[j,1]     #of each triangle
            #Logic to check whether the cebtroid of each triangle is inside the rectangle formed above
            Mat1 = np.array([[x1, y1, 1], [x2, y2, 1], [trianglecoordinate[0,0], trianglecoordinate[0,1], 1]])
            Mat2 = np.array([[x1, y1, 1], [x3, y3, 1], [trianglecoordinate[0, 0], trianglecoordinate[0, 1], 1]])
            Mat3 = np.array([[x3, y3, 1], [x4, y4, 1], [trianglecoordinate[0, 0], trianglecoordinate[0, 1], 1]])
            Mat4 = np.array([[x2, y2, 1], [x4, y4, 1], [trianglecoordinate[0, 0], trianglecoordinate[0, 1], 1]])
            A1 = abs(0.5 * np.linalg.det(Mat1))
            A2 = abs(0.5 * np.linalg.det(Mat2))
            A3 = abs(0.5 * np.linalg.det(Mat3))
            A4 = abs(0.5 * np.linalg.det(Mat4))
            A = 10 * math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
            Atot = A1 + A2 + A3 + A4

#If the triangle is inside then stores the coordinates of the vertices of the triangle in street variable and appends all the triangles one by one to streettriangle matrix
            if Atot > A*0.999999 and Atot < A*1.000001:
                street =np.array([j,your_mesh.x[j,0],your_mesh.y[j,0],your_mesh.x[j,1],your_mesh.y[j,1],your_mesh.x[j,2],your_mesh.y[j,2]])     
                streettriangle.append(street)

    elif file.shapeid[i] < grid_list[k]:
        continue
#The section below generates all stl streets and empties the streettriangle matrix for the next iteration (shape-id)
    else:
        rows = len(streettriangle)
        generate = np.zeros(rows, dtype=mesh.Mesh.dtype)
        for l in range(rows):
            generate['vectors'][l] = np.array([[streettriangle[l][1], streettriangle[l][2]],
                                               [streettriangle[l][3], streettriangle[l][4]],
                                               [streettriangle[l][5], streettriangle[l][6]]])
        generate_street = mesh.Mesh(generate)
        print("Street ",k)
        generate_street.save('Street.stl')
        streettriangle =[]
        if k<(grid_list.ndarray.size-1):
            k=k+1
        else:
            break
            
