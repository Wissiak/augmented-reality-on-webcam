'''
Refer to README.md for usage information

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
The code base for feature dection, feature matching, homography 
decomposition and object projection has been provided as course material 
to the course "Image Processing and Computer Vision 2" taught at OST 
university and has been enhanced with a solver for focal lengths, 
applying homographic projection to the reference image and 
adapted to work with video input.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

@authors Patrick Wissiak, basic material by Martin Weisenhorn
@date May 2023
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
from scipy.optimize import least_squares
import os
from dotenv import load_dotenv
from distutils.util import strtobool

###############################################
# Adaptable parameters:
#   VERBOSE: Print elapsed time per method (monitoring)
#   use_webcam: Use webcam or video input
#   input_video: Filename of video input
#   ref_img_name: Filename of reference image (must be planar object)
#   engineering_method_active: If engineering method should be shown
#   pose_estimation_active: If pose estimation should be shown
#   matches_active: If matches should be shown
#   x_frame: Corner points of reference image (see notes ar_webcam.md for details)
###############################################

load_dotenv()

VERBOSE = strtobool(os.getenv('VERBOSE', 'False'))
use_webcam = strtobool(os.getenv('use_webcam', 'False'))
input_video = os.getenv('input_video') or 'videos/reference-1.mov'
ref_img_name = os.getenv('ref_img_name') or 'images/book1-reference-cut.png'
engineering_method_active = strtobool(os.getenv('engineering_method_active', 'False'))
strict_method_active = strtobool(os.getenv('strict_method_active', 'False'))
pose_estimation_active = strtobool(os.getenv('pose_estimation_active', 'False'))
matches_active = strtobool(os.getenv('matches_active', 'False'))

if 'book1' in ref_img_name:
    x_frame = np.float32([[0, 0], [205, 0], [205, 285], [0, 285]]) # order: top left, top right, bottom right, bottom left

###############################################

# Read reference image
reference_image = cv.imread(ref_img_name)
h, w = reference_image.shape[:2]
fig = plt.figure('Reference Image')
plt.imshow(cv.cvtColor(reference_image, cv.COLOR_BGR2RGB))
plt.show()

edge_colors = [
    (0,0,0),(0,0,0),(0,0,0),(0,0,0),
    (0,0,0),(0,0,0),(0,0,0),(0,0,0),
    (0,0,0),(0,0,0),(0,0,0),(0,0,0),
    (0,0,255),(0,255,0),(255,0,0)
]

# Define a cube as the reference object
ar_object = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]).T

# Lines on the cuboid as sequence of tuples containing the indices of the starting point and the endpoint
edges = [
    [4, 5], [5, 6], [6, 7], [7, 4],  # Lines of back plane
    [0, 4], [1, 5], [2, 6], [3, 7], # Lines connecting front with back-plane
    [0, 1], [1, 2], [2, 3], [3, 0],  # Lines of front plane
    [0, 8], [0, 9], [0, 10],  # Lines indicating the coordinate frame
]

# Scale virtual object
Dx = 60
Dy = 60
Dz = 60
object_points = np.diag((Dx, Dy, Dz)) @ ar_object

# Open webcam or load video input
if use_webcam:
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(input_video)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# try to get the first frame
if cap.isOpened():
    rval, video_frame = cap.read()
else:
    print("Could not open camera")
    exit(0)

M, N = video_frame.shape[0:2]
sift = cv.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.001,
    edgeThreshold=20,
    sigma=1.5,
    nOctaveLayers=4
)

# Brute force matching
matcher = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)

'''
Input
    H_c_b: homography from a planar object in b-coordinates to an image c-coordinates with the ORIGIN on the IMAGE CENTER!
Output
    R_c_b: Rotation matrix
    t_c_cb: Translation vector
    fx, fy: focal lenghts 
'''
def recoverRigidBodyMotionAndFocalLengths(H_c_b):
    a = H_c_b
    Ma = np.array([[a[0,0]**2, a[1,0]**2, a[2,0]**2],
                [a[0,1]**2, a[1,1]**2, a[2,1]**2],
                [a[0,0]*a[0,1], a[1,0]*a[1,1], a[2,0]*a[2,1]]])
    y = np.array([[1], [1], [0]])
    diags = np.linalg.inv(Ma) @ y

    if diags[0] * diags[2] and diags[1] * diags[2]:
        LambdaInv = np.diag(np.sqrt(diags.ravel()))
        B = LambdaInv @ H_c_b
        rx = B[:, [0]]
        ry = B[:, [1]]
        rz = np.cross(rx.ravel(), ry.ravel()).reshape((3,1))
        R_c_b = np.hstack((rx, ry, rz))
        t_c_cb = B[:,[2]]
        fx = LambdaInv[2, 2] / LambdaInv[0, 0]
        fy = LambdaInv[2, 2] / LambdaInv[1, 1]
        return R_c_b, t_c_cb, fx, fy
    else:
        return np.nan, np.nan, np.nan, np.nan

'''
Camera point mapping
'''
def homographyFrom4PointCorrespondences(x_d, x_u):
    # Matrix D und Vektor y
    A = np.zeros((8, 8))
    y = np.zeros((8,))
    for n in range(4):
        A[2*n, :] = [x_u[n, 0], x_u[n, 1], 1, 0, 0, 0, -x_u[n, 0]*x_d[n, 0], -x_u[n , 1]*x_d[n, 0]]
        A[2*n+1, :] = [0, 0, 0, x_u[n, 0], x_u[n, 1], 1, -x_u[n, 0]*x_d[n, 1], -x_u[n, 1]*x_d[n, 1]]
        y[2*n] = x_d[n, 0]
        y[2*n+1] = x_d[n, 1]

    # Compute coefficient vector theta = [a b c ...  h]
    theta = np.linalg.solve(A, y)

    # Compute the homography that maps points from the undistorted image to the distorted image
    H_d_u = np.ones((3, 3))
    H_d_u[0, :] = theta[0:3]
    H_d_u[1, :] = theta[3:6]
    H_d_u[2, 0:2] = theta[6:8]
    return H_d_u

'''
Decompose the homography into rotation matrix, translation vector and focal lengths.
Due to noisy measurements we are using a least squares solution to obtain an optimal solution.
Input
    x_d_center: Starting point of image center for homography decomposition
    x_d: Distorted points of object in the actual image
    x_u: Undistorted corner points of reference object
Output
    R_c_b: Rotation matrix
    t_c_cb: Translation vector
    K_c: Camera intrinsics
'''
def findPoseTransformationParamsLeastSquares(shape, x_d, x_u):
    x_d_center = np.array((shape[0]/2, shape[1]/2))
    # Define the quadratic function to fit
    def loss_func(params):
        x_d_center = params
        cH_c_b = homographyFrom4PointCorrespondences(x_d - x_d_center, x_u)
        _, _, fx, fy = recoverRigidBodyMotionAndFocalLengths(cH_c_b)
        return 1 - fx / fy
    
    try:
        res = least_squares(loss_func, x_d_center)
    except:
        return None, None, None
    x_opt, y_opt = res.x
    if VERBOSE: print(f"Simple LS method - cost: {res.cost}")
    cH_c_b = homographyFrom4PointCorrespondences(x_d - [x_opt, y_opt], x_u)
    R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(cH_c_b)
    K_c = np.array([[fx, 0, x_opt], [0, fy, y_opt], [0, 0, 1]])

    return R_c_b, t_c_cb, K_c


initial_params = [
    0, 0, 0, 0, 0, 0, 1
]
'''
This function tries to minimize the reprojection error between the distorted image
points from the object in the scene and the undistorted image points of the reference
image by using Levenberg-Marquardt 
'''
def findPoseTransformationParamsExactCorrespondence(shape, x_d, x_u):
    global initial_params
    # Fit for x_d_center
    def loss_func(params):
        f = params[0]
        K_c = np.array([
            [f, 0, shape[0]//2],
            [0, f, shape[1]//2],
            [0, 0, 1],
        ])
        # 6 DoF: 
        # - Rotation in x direction
        # - Rotation in y direction
        # - Rotation in z direction
        # - Tranlsation in x direction
        # - Tranlsation in y direction
        # - Tranlsation in z direction

        # Define a rotation vector as rotation axis        
        rodrigues_vector = np.array([params[1], params[2], params[3]])
        R = cv.Rodrigues(rodrigues_vector)[0]

        t_x = params[4]
        t_y = params[5]
        t_z = params[6]

        H = K_c @ np.hstack((R[:, 0:2],[[t_x], [t_y], [t_z]]))

        x_u_homogeneous = np.hstack((x_u, [[1],[1],[1],[1]])).T

        x_uT_homogeneous = H @ x_u_homogeneous
        x_uT = x_uT_homogeneous[0:2,:] / x_uT_homogeneous[2,:]

        return np.ravel(x_d.T - x_uT)
    
    try:
        res = least_squares(loss_func, initial_params, method='lm')
    except Exception as e: 
        print(e)
        return None, None, None
    f_opt = res.x[0]
    if VERBOSE: print(f"\"Exact\" method - cost: {res.cost}")
    if res.cost > 10:
        initial_params = [
            0, 0, 0, 0, 0, 0, 1
        ]
        return None, None, None
    initial_params = res.x # Overwrite starting values for next iteration

    K_c = np.array([
        [f_opt, 0, shape[0]//2],
        [0, f_opt, shape[1]//2],
        [0, 0, 1],
    ])

    R_c_b = cv.Rodrigues(res.x[1:4])[0].reshape(3,3)
    t_c_cb = np.vstack(res.x[4:])

    return R_c_b, t_c_cb, K_c

'''
This method tries to minimize the difference between the two obtained
focal lengths. It does so by trying out pixels around one of the corners 
of the image points of the object in the scene. 
'''
def findPoseTransformationParams(x_d_center, x_d, x_u):
    # Estimate the homography from the body planar surface to the image coordinates with the origin in the center   
    ratio = []
    solution = []
    angle = []
    iter = 0
    rotations = 3
    steps_per_rotation = 50
    delta_per_rotation = 6

    # Try to find an optimal solution by trying out pixels in a spiral around the noisy edge
    for iter in range(rotations * steps_per_rotation):
        alpha = iter * 2 * np.pi / steps_per_rotation
        dr = iter / steps_per_rotation * delta_per_rotation 
        dx = dr * np.cos(alpha)
        dy = dr * np.sin(alpha)
        x_ds = x_d.copy()
        x_ds[1, :] = x_ds[1, :] + np.array((dx, dy))
        cH_c_b = homographyFrom4PointCorrespondences(x_ds - x_d_center, x_u)
        # Determine the pose and the focal lengths
        R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(cH_c_b)
        if not np.isnan(fx):
            ratio.append(min(fx,fy) / max(fx,fy))
            angle.append(alpha)
            solution.append({'R_c_b' : R_c_b, 't_c_cb' : t_c_cb, 'fx' : fx, 'fy' : fy})

    if len(ratio) == 0:
        print("Could not find a Rotation Matrix and Transformation Vector")
        return None, None, None
    
    # Identify the most plausible solution
    ratio = np.array(ratio)
    idx = np.argmax(ratio)
    sol = solution[idx]
    R_c_b = sol['R_c_b']
    t_c_cb = sol['t_c_cb']
    fx = sol['fx']
    fy = sol['fy']

    # Compose the camera intrinsic matrix
    cx = x_d_center[0]
    cy = x_d_center[1]
    K_c = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return R_c_b, t_c_cb, K_c

'''
Transform the object reference to the object in the cluttered scene
'''
def show_pose_estimation(H):
    # Remove //2 if PyrDown is not used
    resultimage = cv.warpPerspective(reference_image, H, (N//2, M//2))
    cv.imshow("Pose Estimation", resultimage)

'''
Monitor elapsed time between start_time and current time
'''
def print_time(job, start_time):
    if not VERBOSE: return
    elapsed_time = time.time() - start_time
    print(f"{job} took {elapsed_time}s")
    return time.time()

'''
Show matches, starting with the most reliable
'''
def show_matches(title, matches, reference_image, reference_keypoints, video_frame, frame_keypoints):
    sorted_matches = sorted(matches, key = lambda x:x.distance)
    plt_img = cv.drawMatches(video_frame, frame_keypoints, reference_image, reference_keypoints, sorted_matches[:400], video_frame, flags=2)
    plt_img = cv.resize(plt_img, (plt_img.shape[1]//2, plt_img.shape[0]//2)) 
    cv.imshow(title, plt_img)

'''
Draw the AR object to the given video frame
'''
def draw_ar_object(video_frame, K_c, R_c_b, t_c_cb):
    if R_c_b is not None:
        # Project virtual object onto reference plane
        points_c = R_c_b @ object_points + t_c_cb
        image_points_homogeneous = (K_c @ points_c)
        image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

        image_points = np.int64(image_points)
        for edge_id, edge in enumerate(edges):
            pt1 = image_points[:, edge[0]]
            pt2 = image_points[:, edge[1]]
            cv.line(img=video_frame, pt1=(pt1[1], pt1[0]), pt2=(pt2[1], pt2[0]), color=edge_colors[edge_id], thickness=3, lineType=8,)
    return video_frame

def webcam_ar(video_frame):
    video_frame1 = video_frame.copy()
    if engineering_method_active:
        video_frame2 = video_frame.copy()
    if strict_method_active:
        video_frame3 = video_frame.copy()
    start_time = time.time()

    # Compute descriptors and keypoints
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
    frame_keypoints, frame_descriptors = sift.detectAndCompute(video_frame, None)

    start_time = print_time("SIFT computation", start_time)

    # 3. Match keypoints and show the matches
    matches = matcher.match(frame_descriptors, reference_descriptors)
    if VERBOSE: print('{} matches found'.format(len(matches)))

    start_time = print_time("Finding matches", start_time)

    if matches_active:
        show_matches('Brute Force Matching Result', matches, reference_image, reference_keypoints, video_frame1, frame_keypoints)

    # Fit the homography model into the found keypoint correspondences robustly and get a mask of inlier matches:
    dstPtsCoords = np.float32([frame_keypoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
    srcPtsCoords = np.float32([reference_keypoints[m.trainIdx].pt for m in matches]).reshape(-1,2)
    H, mask = cv.findHomography(srcPoints=srcPtsCoords, dstPoints=dstPtsCoords, method=cv.RANSAC, ransacReprojThreshold=5.0)
    # H, mask = cv.estimateAffinePartial2D(from_=srcPtsCoords, to=dstPtsCoords, inliers=None, method=cv.RANSAC, ransacReprojThreshold=25.0)

    start_time = print_time("Calculating homography", start_time)

    matches_mask = mask.ravel().tolist()

    reliable_matches_indices = np.nonzero(matches_mask)[0]
    no_of_matches = len(reliable_matches_indices)
    if VERBOSE: print(f"Number of stable matches: {no_of_matches}")

    if matches_active:
        reliable_matches = np.array(matches)[reliable_matches_indices]
        show_matches('Reliable Matches', reliable_matches, reference_image, reference_keypoints, video_frame, frame_keypoints)

    if no_of_matches > 50:

        # Get projections of corners of reference object in cluttered scene
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        transformed_corners = cv.perspectiveTransform(corners, H)

        # Draw boundaries of reference object on cluttered scene
        cv.polylines(video_frame1, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv.LINE_AA)

        # Create homography from undistorted image (in pseudo real world dimensions) to corners of reference object in cluttered scene
        x_corners = np.concatenate((transformed_corners[:,:,1], transformed_corners[:,:,0]), axis=1) # top left, top right, bottom right, bottom left

        x_d_center = np.array((video_frame.shape[0]/2, video_frame.shape[1]/2))
        start_time = time.time()

        R_c_b, t_c_cb, K_c = findPoseTransformationParamsLeastSquares(video_frame.shape, x_corners, x_frame)
        start_time = print_time("Least Squares Method", start_time)
        video_frame1 = draw_ar_object(video_frame1, K_c, R_c_b, t_c_cb)

        if strict_method_active:
            start_time = time.time()
            R_c_b, t_c_cb, K_c = findPoseTransformationParamsExactCorrespondence(video_frame3.shape, x_corners, x_frame)
            start_time = print_time("Strict LS Method", start_time)
            video_frame3 = draw_ar_object(video_frame3, K_c, R_c_b, t_c_cb)


        if engineering_method_active:
            start_time = time.time()
            # To show the time difference between the two methods:
            R_c_b, t_c_cb, K_c = findPoseTransformationParams(x_d_center, x_corners, x_frame)
            start_time = print_time("Engineering Method", start_time)
            video_frame2 = draw_ar_object(video_frame2, K_c, R_c_b, t_c_cb)

    cv.imshow("Least Squares Method", video_frame1)
    if engineering_method_active:
        cv.imshow("Engineering Method", video_frame2)
    if strict_method_active:
        cv.imshow("Strict LS Method", video_frame3)
    
    if pose_estimation_active:
        # To show the transformed reference image based on the homography transformation:
        show_pose_estimation(H)

if use_webcam:
    while rval:
        rval, video_frame = cap.read()

        video_frame = cv.pyrDown(video_frame, dstsize=(N // 2, M // 2))

        key = cv.waitKey(1)
        if key == ord('q'): # exit on Q
            break

        webcam_ar(video_frame)
else:
    for fno in range(0, total_frames, 3):
        cap.set(cv.CAP_PROP_POS_FRAMES, fno) # Speed up video mode
        rval, video_frame = cap.read()

        if not rval:
            break

        video_frame = cv.pyrDown(video_frame, dstsize=(N // 2, M // 2))

        key = cv.waitKey(1)
        if key == ord('q'): # exit on Q
            break

        webcam_ar(video_frame)

cap.release()
cv.destroyAllWindows()
