# face_swapping

* Load the two images and convert them into grayscale format
* Detect the first face and extract the facial landmarks using a pre-trained facial landmark detector offered by the dlib library. This detector is used to estimate the location of 68 (x, y) coordinates that map to facial structures on the face. You need to download 'shape_predictor_68_face_landmarks.dat' to initialize the face predictor.

The indexes of the 68 coordinates can be visualized on the image below:
![alt text](https://github.com/Mariem-sicom/face_swapping/blob/main/facial_landmarks_68markup.jpg?raw=true)

* Find the convex hull of the facial landmarks, apply it on a mask and put that mask on the first image to extract the face
Face segmentation: split the first face into triangles using Delaunay Triangulation (we use this method to keep all the proportions so all the expressions will be as natural as possible)

Delaunay Triangulation: get the rectangle where we want to find the triangles (the rectangle surrounding the convex hull) and get the triangles list

* To match the two faces we will look for the indexes (of the 68 coordinates) of each traingle on the first image to know what specific landmarks points each triangle connect
* Connect the landmarks in the second face to form the same triangles
* Select the corresponfing triangles and wrap them so that the triangles of the first image match exactly the second image in shape and size : find the rectangle surrounding each triangle, create a mask to extract the triangle, use the mask to extract only the triangle, wrap the triangles(transform the first triangle so it can fit the triangle of the second image using an affine transformation)
* Swap both faces: overlapp the extracted wraped triangles, create a mask to extract everything in the second image exept the face, put the warped face on the extracted background, adjust the colors
