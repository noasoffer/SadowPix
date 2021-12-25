# ShadowPix

 Implementation of "SHADOWPIX: Multiple Images from Self Shadowing".

By creating different height surfaces that cast shadows on each other, a number of images can be created depending on the direction of the light.

As part of the project in the course: Algorithms for Modeling, Fabrication and Printing of 3D Objects, we implemented the two algorithms that described in the paper.


Input (image)            |  Output (mesh)
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/74652585/147386568-12aa747b-25b7-4045-9216-0ac735910255.png) | ![image](https://user-images.githubusercontent.com/74652585/147386574-8efe6e5c-1deb-4157-854d-7597f7b241aa.png)
![image](https://user-images.githubusercontent.com/74652585/147386589-18ae77d9-d8c9-4941-9dc4-bb5dd07572dd.png) | ![image](https://user-images.githubusercontent.com/74652585/147386595-5814307e-19a1-4251-a155-90f11b8632bb.png)
!![image](https://user-images.githubusercontent.com/74652585/147386934-80561cf3-b124-4ac2-a013-c31acc25860f.png) | ![image](https://user-images.githubusercontent.com/74652585/147386942-172649ec-e45c-4ab2-b650-0f55daa45978.png)

For simulating the mesh we used:Spin 3D Mesh Converter Software

## Local Method 

**run:**
```js
❯ python local_method.py
```
Mesh will be saved in to mesh_local.obj file. 

## Global Method 

**run:**
```js
❯ python global_method.py 
# Convergence might take some time
```
Mesh will be saved in to mesh_global.obj file. 


