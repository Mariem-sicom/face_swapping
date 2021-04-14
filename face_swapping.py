import cv2
import numpy as np
import dlib

# load the two images and convert them into grayscale format
img1 = cv2.imread('face1.jpg')
print(img1.shape[:])
img1_copy = np.copy(img1)
img1Gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread('face2.png')
print(img2.shape[:])
img2Gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img2_new_face = np.zeros_like(img2)

# detect the face and extract the facial landmarks
detector = dlib.get_frontal_face_detector()  # Initialize dlib's face detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Initialize dlib's shape predictor

faces1 = detector(img1Gray, 1)  # Detecting faces in the grayscale image
print (len(faces1), 'face detected')
for face in faces1:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(img1_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)


for face in faces1:
    landmarks = predictor(img1Gray, face)  # Get the shape
    facial_landmarks = []
    for i in range(0, 68):
        x = landmarks.part(i).x  # Extract coordinates of the facial landmarks
        y = landmarks.part(i).y
        facial_landmarks.append((x, y))
        cv2.circle(img1_copy, (x, y), 2, (0, 255, 0), -1)  # -1 to fill the circle


# find the convex hull of the facial landmarks and apply it on a mask
points = np.array(facial_landmarks, np.int32)
hull = cv2.convexHull(points)
mask = np.zeros(img1Gray.shape[:2], np.uint8)
cv2.fillConvexPoly(mask, hull, 255)
face_img = cv2.bitwise_and(img1, img1, mask=mask)

# Delaunay Triangulation
rect = cv2.boundingRect(hull)  # get the rectangle surrounding the convex hull
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(facial_landmarks)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.uint32)
# for t in triangles:
#     pt1 = (t[0], t[1])
#     pt2 = (t[2], t[3])
#     pt3 = (t[4], t[5])
#     cv2.line(img1_copy, pt1, pt2, (0, 0, 255), 1)
#     cv2.line(img1_copy, pt2, pt3, (0, 0, 255), 1)
#     cv2.line(img1_copy, pt1, pt3, (0, 0, 255), 1)

# Matching the two faces triangulation


def get_index(ndarray):
    index = None
    for ind in ndarray[0]:
        index = ind
        break
    return index


# Find the landmarks points  each triangle on the 1st face connects
indexes_triangles = []
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    indexes_triangle = []
    index_pt1 = np.where((points == pt1).all(axis=1))  # returns (array([58], dtype=int64),)
    index_pt1 = get_index(index_pt1)  # returns 58
    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = get_index(index_pt2)
    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = get_index(index_pt3)
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle_indexes = [index_pt1, index_pt2, index_pt3]
    indexes_triangles.append(triangle_indexes)

# Extract the facial landmarks from the second image
faces2 = detector(img2Gray, 1)
for face in faces2:
    facial_landmarks2 = []
    landmarks2 = predictor(img2Gray, face)
    for i in range(0, 68):
        x = landmarks2.part(i).x
        y = landmarks2.part(i).y
        facial_landmarks2.append((x, y))
        # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)

# Determine the landmarks coordinates on the second image
for triangle in indexes_triangles:
    img1_pt1 = facial_landmarks[triangle[0]]
    img1_pt2 = facial_landmarks[triangle[1]]
    img1_pt3 = facial_landmarks[triangle[2]]
    # cv2.line(img1, img1_pt1, img1_pt2, (0, 0, 255), 1)
    # cv2.line(img1, img1_pt2, img1_pt3, (0, 0, 255), 1)
    # cv2.line(img1, img1_pt1, img1_pt3, (0, 0, 255), 1)
    cropped_triangle1 = np.array([img1_pt1, img1_pt2, img1_pt3], np.int32)
    rect1 = cv2.boundingRect(cropped_triangle1)
    (x, y, h, w) = rect1
    cropped_rect1 = img1[y:y+w, x:x+h]
    cropped_mask1 = np.zeros(cropped_rect1.shape[:2], np.uint8)
    # Find the coordinates of the points in the cropped rect
    points_tr1 = np.array([[img1_pt1[0] - x, img1_pt1[1] - y],
                           [img1_pt2[0] - x, img1_pt2[1] - y],
                           [img1_pt3[0] - x, img1_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_mask1, points_tr1, 255)
    cropped_rect1 = cv2.bitwise_and(cropped_rect1, cropped_rect1, mask=cropped_mask1)

    img2_pt1 = facial_landmarks2[triangle[0]]
    img2_pt2 = facial_landmarks2[triangle[1]]
    img2_pt3 = facial_landmarks2[triangle[2]]
    # cv2.line(img2, img2_pt1, img2_pt2, (0, 0, 255), 1)
    # cv2.line(img2, img2_pt2, img2_pt3, (0, 0, 255), 1)
    # cv2.line(img2, img2_pt1, img2_pt3, (0, 0, 255), 1)
    cropped_triangle2 = np.array([img2_pt1, img2_pt2, img2_pt3], np.int32)
    rect2 = cv2.boundingRect(cropped_triangle2)
    (x, y, h, w) = rect2
    cropped_rect2 = img2[y:y+w, x:x+h]
    cropped_mask2 = np.zeros(cropped_rect2.shape[:2], np.uint8)
    # Find the coordinates of the points in the cropped rect
    points_tr2 = np.array([[img2_pt1[0] - x, img2_pt1[1] - y],
                           [img2_pt2[0] - x, img2_pt2[1] - y],
                           [img2_pt3[0] - x, img2_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_mask2, points_tr2, 255)
    cropped_rect2 = cv2.bitwise_and(cropped_rect2, cropped_rect2, mask=cropped_mask2)

    # Warp triangles
    points_tr1 = np.float32(points_tr1)
    points_tr2 = np.float32(points_tr2)
    M = cv2.getAffineTransform(points_tr1, points_tr2)
    warped_triangle1 = cv2.warpAffine(cropped_rect1, M, (h, w))

    # Reconstruct destination face
    img2_new_face_area = img2_new_face[y:y+w, x:x+h]
    img2_new_face_area = cv2.add(img2_new_face_area, warped_triangle1)
    img2_new_face[y:y+w, x:x+h] = img2_new_face_area


# Face swapped (putting 1st face into 2nd face)
points2 = np.array(facial_landmarks2, np.int32)
convexhull2 = cv2.convexHull(points2)
img2_face_mask = np.zeros_like(img2Gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)


img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)


cv2.imshow("first face", img1)
cv2.imshow("second face", img2)
cv2.imshow("face swapping", seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()



