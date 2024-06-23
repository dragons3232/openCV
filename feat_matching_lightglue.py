import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd


def draw_matches(
    img1, kpts1, img2, kpts2, matches, line_color=(0, 255, 0), circle_color=(0, 0, 255)
):

    # Create an output image with the two input images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1 : w1 + w2] = img2

    # Draw lines between matching keypoints
    for i1, i2 in matches:
        x1, y1 = kpts1[i1].ravel()
        x2, y2 = kpts2[i2].ravel()

        cv2.circle(out_img, (int(x1), int(y1)), 3, circle_color, -1)
        cv2.circle(out_img, (int(x2) + w1, int(y2)), 3, circle_color, -1)
        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + w1, int(y2)), line_color, 1)

    return out_img



source1 = "p1.jpg"
source2 = "cropped.jpg"

# SuperPoint+LightGlue
device = 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)  # load the matcher

# # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features="disk").eval().to(device)  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image(source1).to(device)
image1 = load_image(source2).to(device)

img1 = cv2.imread(source1)
img2 = cv2.imread(source2)

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension
matches = matches01["matches"]  # indices with shape (K,2)
points0 = feats0["keypoints"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1["keypoints"][matches[..., 1]]  # coordinates in image #1, shape (K,2)

matchesZipped = list(zip(range(len(points0)), range(len(points1))))

out_img = draw_matches(img1, points0, img2, points1, matchesZipped)
cv2.imshow('out image', out_img)
cv2.waitKey(0)