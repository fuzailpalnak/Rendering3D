import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Load reference image and convert it to gray scale
from task1.src.ops import ransac
from task1.src.pose import dlt_ransac
from task1.src.util import draw_pairs, draw_key_points

referenceImage = cv2.imread(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/wetransfer_img_3449-mov_2022-11-08_2048/image00001.jpeg", 0)


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == "vn":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == "vt":
                self.texcoords.append(list(map(float, values[1:3])))
            # elif values[0] in ('usemtl', 'usemat'):
            # material = values[1]
            # elif values[0] == 'mtllib':
            # self.mtl = MTL(values[1])
            elif values[0] == "f":
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split("/")
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                # self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))


# Show image
plt.imshow(referenceImage, cmap="gray")
plt.show()

# Load the source image and convert it to gray scale
sourceImage = cv2.imread(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/source_test_1.jpg", 0)

# Show image
plt.imshow(sourceImage, cmap="gray")
plt.show()


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
referenceImagePts = orb.detect(referenceImage, None)
sourceImagePts = orb.detect(sourceImage, None)

# compute the descriptors with ORB
referenceImagePts, referenceImageDsc = orb.compute(referenceImage, referenceImagePts)
sourceImagePts, sourceImageDsc = orb.compute(sourceImage, sourceImagePts)

#  Paint the key points over the original image
referenceImageFeatures = cv2.drawKeypoints(
    referenceImage, referenceImagePts, referenceImage, color=(0, 255, 0), flags=0
)
sourceImageFeatures = cv2.drawKeypoints(
    sourceImage, sourceImagePts, sourceImage, color=(0, 255, 0), flags=0
)

MIN_MATCHES = 30

# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Compute model keypoints and its descriptors
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)

# Compute scene keypoints and its descriptors
sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)

# Match frame descriptors with model descriptors
matches = bf.match(referenceImageDsc, sourceImageDsc)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

point_map = np.array(
    [
        [
            referenceImagePts[match.queryIdx].pt[0],
            referenceImagePts[match.queryIdx].pt[1],
            sourceImagePts[match.trainIdx].pt[0],
            sourceImagePts[match.trainIdx].pt[1],
        ]
        for match in matches
    ]
)


def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)

    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)


# HOMOGRAPHY ESTIMATION
pm, matched_pairs = dlt_ransac(point_map)
camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

sourcePoints = np.float32([referenceImagePts[m.queryIdx].pt for m in matches]).reshape(
    -1, 1, 2
)
destinationPoints = np.float32(
    [sourceImagePts[m.trainIdx].pt for m in matches]
).reshape(-1, 1, 2)
homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)

matchesMask = mask.ravel().tolist()

drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None,
                      matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(referenceImage, referenceImagePts, sourceImage,
                         sourceImagePts, matches, None, **drawParameters)

plt.figure(figsize=(12, 6))
plt.imshow(result, cmap='gray')
plt.show()

homo, _ = ransac(point_map)
pm1 = projection_matrix(camera_parameters, homo)

mapping_img = draw_key_points(
    referenceImage, sourceImage.copy(), list(matched_pairs), pairs=matched_pairs
)
plt.figure(figsize=(12, 6))
plt.imshow(mapping_img, cmap="gray")
plt.show()

obj = OBJ(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/chair.obj", swapyz=True)

# project cube or model
def render(img, obj, projection, dlt, model, color=False):

    vertices = obj.vertices
    scale_matrix = np.eye(3) * 6
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        points = 20 * np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        pp = np.c_[points, np.ones(len(points))]
        # uv2 = np.dot(dlt, pp.T)
        # uv2 = uv2 / uv2[2, :]
        #
        projections = np.zeros((points.shape[0], 3))
        for i in range(points.shape[0]):
            projections[i, :] = np.matmul(dlt, np.transpose(pp[i, :]))
            projections[i, :] = projections[i, :] / projections[i, 2]

        imgpts = []
        for aa in projections:
            imgpts.append([int(aa[0]), int(aa[1])])
        imgpts = np.int32(np.array(imgpts)).reshape(-1, 2)

        # dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        # imgpts = np.int32(dst).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)



        # cv2.fillConvexPoly(img, imgpts, (80, 27, 211))
        return img
    return img


frame = render(cv2.imread(r"/home/palnak/Workspace/Studium/msc/sem3/assignment/AR/task1/data/source_test_1.jpg"), obj, pm1, pm, referenceImage, False)
plt.figure(figsize=(12, 6))
plt.imshow(frame, cmap="gray")
plt.show()
