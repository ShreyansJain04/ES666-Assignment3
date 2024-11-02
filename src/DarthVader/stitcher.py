
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def find_matches(img1, img2, max_matches=30):
  
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_keypoints = []
    for match in matches:
        matched_keypoints.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))

    src_pts = np.float32([kp[0] for kp in matched_keypoints[:max_matches]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[1] for kp in matched_keypoints[:max_matches]]).reshape(-1, 1, 2)

    return np.array(matched_keypoints[:max_matches]), src_pts, dst_pts

def estimate_homography(keypoint_pairs):

    A_matrix = []
    for kp1, kp2 in keypoint_pairs:
        x1, y1 = kp1
        x2, y2 = kp2
        A_matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A_matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    A_matrix = np.asarray(A_matrix)
    _, _, Vh = np.linalg.svd(A_matrix)
    H = Vh[-1, :] / Vh[-1, -1]
    return H.reshape(3, 3)


def ransac_homography(correspondences, iterations=3000, error_threshold=2, sample_size=4):

    best_H = None
    best_inliers = []

    for _ in tqdm(range(iterations)):
        sampled_indices = np.random.choice(len(correspondences), sample_size, replace=False)
        sample = correspondences[sampled_indices]
        H = estimate_homography(sample)
        inliers = []

        for kp1, kp2 in correspondences:
            src = np.append(kp1, 1)
            dst = np.append(kp2, 1)
            estimated_dst = np.dot(H, src)
            estimated_dst /= estimated_dst[-1]
            error = np.linalg.norm(dst - estimated_dst)
            if error < error_threshold:
                inliers.append((kp1, kp2))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    print(f"Max inliers = {len(best_inliers)}")
    return best_H, best_inliers


def calculate_warp_bounds(H, width, height):
    """Calculates the bounding box of the warped image."""
    corners = np.array([[0, width - 1, 0, width - 1],
                       [0, 0, height - 1, height - 1],
                       [1, 1, 1, 1]])
    transformed_corners = np.dot(H, corners)
    transformed_corners /= transformed_corners[2, :]
    x_min = int(np.min(transformed_corners[0]))
    x_max = int(np.max(transformed_corners[0]))
    y_min = int(np.min(transformed_corners[1]))
    y_max = int(np.max(transformed_corners[1]))
    return x_min, x_max, y_min, y_max



def warp_image(source, H, target, use_forward=False, offset=(2300, 800)):
    """Warps the source image onto the destination image."""

    h, w, _ = source.shape
    H_inv = np.linalg.inv(H)


    if use_forward:
        coords = np.indices((w, h)).reshape(2, -1)
        homogeneous_coords = np.vstack((coords, np.ones(coords.shape[1])))
        transformed_coords = np.dot(H, homogeneous_coords)
        transformed_coords /= transformed_coords[2, :]

        x_output, y_output = transformed_coords.astype(np.int32)[:2, :]

        valid_indices = (x_output >= 0) & (x_output < target.shape[1]) & \
                        (y_output >= 0) & (y_output < target.shape[0])

        x_output = x_output[valid_indices] + offset[0]
        y_output = y_output[valid_indices] + offset[1]
        x_input = coords[0][valid_indices]
        y_input = coords[1][valid_indices]

        target[y_output, x_output] = source[y_input, x_input]

    else:  # Inverse Mapping
        x_min, x_max, y_min, y_max = calculate_warp_bounds(H, w, h)

        x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        coords = np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(x_coords.size)))

        transformed_coords = np.dot(H_inv, coords)
        transformed_coords /= transformed_coords[2, :]

        x_input = transformed_coords[0].astype(np.int32)
        y_input = transformed_coords[1].astype(np.int32)

        valid = (x_input >= 0) & (x_input < w) & (y_input >= 0) & (y_input < h)
        final_x = x_coords.ravel() + offset[0]
        final_y = y_coords.ravel() + offset[1]
        valid &= (final_x >= 0) & (final_x < target.shape[1]) & \
                 (final_y >= 0) & (final_y < target.shape[0])

        valid_indices = np.where(valid)[0]
        if not valid_indices.size:
            print("No valid coordinates found after applying offset and boundary checks.")
            return
        target[final_y[valid_indices], final_x[valid_indices]] = source[y_input[valid_indices], x_input[valid_indices]]


class PyramidBlender:
    """Blends images using Gaussian and Laplacian pyramids."""

    def __init__(self, levels=6):
        self.levels = levels

    def gaussian_pyramid(self, img):
        """Constructs a Gaussian pyramid."""
        pyramid = [img]
        for _ in range(self.levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def laplacian_pyramid(self, img):
        """Constructs a Laplacian pyramid."""

        pyramid = []
        for _ in range(self.levels - 1):
            down = cv2.pyrDown(img)
            up = cv2.pyrUp(down, dstsize=(img.shape[1], img.shape[0]))
            laplacian = cv2.subtract(img.astype(float), up.astype(float))
            pyramid.append(laplacian)
            img = down
        pyramid.append(img.astype(float))  # Add the last Gaussian level
        return pyramid

    def blend_pyramids(self, lapA, lapB, mask_pyramid):
        """Blends two Laplacian pyramids using a mask."""
        blended_pyramid = []
        for i, mask in enumerate(mask_pyramid):
            mask_3ch = cv2.merge((mask, mask, mask)) # Create 3 channel mask
            blended_level = lapA[i] * mask_3ch + lapB[i] * (1 - mask_3ch)
            blended_pyramid.append(blended_level)
        return blended_pyramid


    def reconstruct(self, pyramid):
        """Reconstructs an image from a Laplacian pyramid."""
        img = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            img = cv2.pyrUp(img, dstsize=level.shape[:2][::-1]).astype(float) + level.astype(float)
        return img

    def create_mask(self, img):
        """Creates a binary mask from an image."""
        mask = np.all(img != 0, axis=2)
        mask_img = np.zeros(img.shape[:2], dtype=float)
        mask_img[mask] = 1.0
        return mask_img

    def blend(self, img1, img2):
        """Blends two images using pyramid blending."""

        lap1 = self.laplacian_pyramid(img1)
        lap2 = self.laplacian_pyramid(img2)

        mask1 = self.create_mask(img1).astype(bool)
        mask2 = self.create_mask(img2).astype(bool)
    
        # Handle potential size differences in masks
        if mask1.shape != mask2.shape:
            min_shape = np.minimum(mask1.shape, mask2.shape)
            mask1 = mask1[:min_shape[0], :min_shape[1]]
            mask2 = mask2[:min_shape[0], :min_shape[1]]

        overlap = mask1 & mask2
        y, x = np.where(overlap)

        if len(x) == 0:
            min_x, max_x = img1.shape[1] // 2, img1.shape[1] // 2
        else:
            min_x, max_x = np.min(x), np.max(x)


        mask = np.zeros(img1.shape[:2])
        mask[:, :(min_x + max_x) // 2] = 1.0

        mask_pyramid = self.gaussian_pyramid(mask)
        blended_pyramid = self.blend_pyramids(lap1, lap2, mask_pyramid)
        blended_img = self.reconstruct(blended_pyramid)

        return blended_img, mask1, mask2


class PanaromaStitcher:
    """Stitches images together to create a panorama."""

    def __init__(self, max_features=30, match_ratio=0.75, ransac_err=2.0, 
                 pyr_levels=6, warp_dim=(600, 400), image_offset=(2300, 800)):
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_err = ransac_err
        self.pyr_levels = pyr_levels
        self.warp_dim = warp_dim
        self.offset = image_offset
        self.img_files = None
        self.scene_id = None


    def make_panaroma_for_images_in(self, path):
        """Creates a panorama from images in a directory."""

        self.img_files = sorted(glob.glob(os.path.join(path, '*')))

        img_count = len(self.img_files)
        print(f"Found {img_count} images for stitching.")
        
        

        self.scene_id = self._get_scene_id(self.img_files[0])
        output_path = f'outputs/scene{self.scene_id}/custom'
        os.makedirs(output_path, exist_ok=True)

        final_H = np.eye(3)  
        

        if img_count == 6:
            final_H = self._stitch_six()
        elif img_count == 5:
            final_H = self._stitch_five()
        elif img_count == 4:
            final_H = self._stitch_four()
        else:
            final_H = self._stitch_three()


        final_panorama = self._blend_all()
        cv2.imwrite(os.path.join(output_path, 'blended_image.png'), final_panorama)
        return final_panorama, final_H
    
    
    def _stitch_six(self):
        """Stitches a six-image scene."""
        H = np.eye(3)
        H = self._stitch_and_save(2, 1, H)  # 2->1
        H = self._stitch_and_save(1, 0, H)  # 1->0

        H = np.eye(3)
        H = self._stitch_and_save(2, 2, H)  # Reference
        H = self._stitch_and_save(2, 3, H)  # 2->3
        H = self._stitch_and_save(3, 4, H)  # 3->4
        H = self._stitch_and_save(4, 5, H)  # 4->5
        return H

    def _stitch_five(self):
        """Stitches a five-image scene."""

        H = np.eye(3)
        H = self._stitch_and_save(2, 1, H)
        H = self._stitch_and_save(1, 0, H)

        H = np.eye(3)
        H = self._stitch_and_save(2, 2, H)
        H = self._stitch_and_save(2, 3, H)
        H = self._stitch_and_save(3, 4, H)
        return H

    def _stitch_four(self):
        """Stitches a four-image scene."""
        H = np.eye(3)
        H = self._stitch_and_save(2, 1, H) # Changed indices
        H = self._stitch_and_save(1, 0, H) # Changed indices

        H = np.eye(3)
        H = self._stitch_and_save(2, 2, H) # Changed indices
        H = self._stitch_and_save(2, 3, H) # Changed indices
        return H

    def _stitch_three(self):
        """Stitches a three-image scene."""
        H = np.eye(3)
        H = self._stitch_and_save(1, 0, H)
        H = self._stitch_and_save(1, 1, H) # 
        return H



    def _stitch_and_save(self, src_index, dst_index, prev_H):
        """Stitches two images and saves the result."""

        canvas = np.zeros((3000, 6000, 3), dtype=np.uint8)

        src_img = cv2.imread(self.img_files[src_index])
        dst_img = cv2.imread(self.img_files[dst_index])

        if src_img is None or dst_img is None:
            print(f"Error: Could not read image {self.img_files[src_index]} or {self.img_files[dst_index]}")
            return prev_H

        print(f"Stitching: {os.path.basename(self.img_files[src_index])} -> {os.path.basename(self.img_files[dst_index])}")

        src_resized = cv2.resize(src_img, self.warp_dim)
        dst_resized = cv2.resize(dst_img, self.warp_dim)

        kp_pairs, src_pts, dst_pts = find_matches(dst_resized, src_resized, max_matches=self.max_features)

        if len(kp_pairs) < 4:
            print("Error: Not enough matches found.")
            return prev_H

        H, _ = ransac_homography(kp_pairs, iterations=3000, error_threshold=self.ransac_err)

        if H is None:
            return prev_H


        # Correctly apply cumulative homography ONLY when warping subsequent images
        if np.array_equal(prev_H, np.eye(3)):  # First image, no previous H
            warp_image(dst_resized, H, canvas, offset=self.offset)
            cumulative_H = H # Initialize cumulative_H
        else:
            cumulative_H = np.dot(prev_H, H)
            warp_image(dst_resized, cumulative_H, canvas, offset=self.offset)


        output_file = f'outputs/scene{self.scene_id}/custom/warped_{dst_index}.png' # unique filenames
        cv2.imwrite(output_file, canvas)
        return cumulative_H



    def _blend_all(self):
        """Blends all warped images."""

        output_dir = f'outputs/scene{self.scene_id}/custom'
        warped_files = sorted(glob.glob(os.path.join(output_dir, 'warped_*.png')))

        if not warped_files:
            raise ValueError("No warped images found for blending.")

        blender = PyramidBlender(levels=self.pyr_levels)
        final_image = cv2.imread(warped_files[0])

        for img_file in warped_files[1:]:
            print(f"Blending: {os.path.basename(img_file)}")
            img2 = cv2.imread(img_file)
            if img2 is not None:
                final_image, _, _ = blender.blend(final_image, img2) # Updated blend method call
        return final_image


    def _get_scene_id(self, file_path):
        """Extracts the scene ID from the file path."""
        dir_name = os.path.basename(os.path.dirname(file_path))
        scene_id = ''.join(filter(str.isdigit, dir_name))
        return int(scene_id) if scene_id.isdigit() else 1

