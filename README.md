# Augmented Reality on Webcam
Show virtual object on plane of reference image without camera calibration. 

This work was done in spring of 2023 after attending the Image Processing and Computer Vision course at OST in Rapperswil, where the underlying algorithms are taught.

**Tags**
Augmented Reality (AR), Wireframe, OpenCV, Pose Estimation, Focal Length Estimation, Homography, Planar Object, SIFT

## Showcase
The following videos show the possibilities of this script:
- [showcase_matches.mov](./videos/showcase_matches.mov) Shows the matches from the reference object to the object in the scene with the matches. An top are all found matches and on the bottom only those filtered by RANSAC.

https://github.com/Wissiak/augmented-reality-on-webcam/assets/31206249/ec211916-f05a-4c6d-add3-14bb69807ecd

- [showcase_pose_estimation.mov](./videos/showcase_pose_estimation.mov) Warps the *reference image* into the pose seen in the video.

https://github.com/Wissiak/augmented-reality-on-webcam/assets/31206249/d1b8dc62-ba4c-4186-9aab-da422fa19456

- [shwocase_least_squares.mov](./videos/shwocase_least_squares.mov) Shows the difference between the engineering and the least squares method.

https://github.com/Wissiak/augmented-reality-on-webcam/assets/31206249/d54ffce7-fff7-4f30-8c22-f5fee5d7cbd3


## Usage and Installation
Use provided conda environment and run `python ar_webcam.py`. 

### Notes 
- RANSAC threshold may not be set too high because wrong matches corrupt the resulting homography.
- x_frame equals the size of the real object. E.g. plane of the reference object is 20.5cm*28.5cm then the x_frame should look like `[[0, 0], [205, 0], [205, 285], [0, 285]]` - format: [x,y]

### Register new reference object
To register a new reference object, do the following steps:
1. Take a picture of the reference object - you can use `image_capture.py` - and set the name 'reference-2' in the console. To take a picture, focus the webcam window and press the key 'y'. This will save the image to `images/reference-2-0.png`. Rename the file to `reference-2.png`.
2. Set fname in `unwarp_image.py` to the location of the image containing the new reference object and start `unwarp_image.py`
3. Hover the corner points of the planar object as given in the underneath image and write down the coordinates.
![](./unwarp_image.png)
4. Set your coordinates in `unwarp_image.py`:
```
if fname == 'images/reference-2':
pts = np.array([
    (x1, y1), # top left
    (x2, y2), # top right
    (x3, y3), # bottom left
    (x4, y4) # bottom right
])
```
5. Restart `unwarp_image.py` which will write the unwarped image to `images/reference-2-cut.png`.
6. In `ar_webcam.py` set `ref_img_name` to `images/reference-2-cut.png` and add the "real world coordinates" of the reference object (see notes above):
```
if ref_img_name == 'images/reference-2-cut.png':
    x_frame = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) # order: top left, top right, bottom right, bottom left
```
7. Start `ar_webcam.py` and verify that the image of your unwarped reference object is shown in the first window.

### Scale the augmented object
The augmented object can be scaled by updating `Dx`, `Dy` and `Dz` in `ar_webcam.py`. For example if you want to show an augmented object that is the same size as the reference object, you can use:
```
Dx = width
Dy = height
Dz = any
```
Where width and height correspond to the "real world coordinates". This will then look like this:
![](./scaled_object.png)

### Pose Estimation
The `warpPerspective` function from opencv can be used to warp the reference object to the estimated pose in the scene:
![](./pose-estimation.png)

To activate the functionality, set `pose_estimation_active=True` in the `.env` file.
