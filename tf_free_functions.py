import time
import numpy as np
import cv2


def fit_to_size(image, target_size, gt_boxes=None, dtype=np.float32, norm=True, num_landmarks=10):
    ih, iw = target_size # output = 416x416
    h, w, _ = image.shape

    # print("image", image.shape)

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    if h * w > ih * iw:
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)


    # image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=dtype)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized

    if norm:
        image_padded = image_padded / 255.

    # print(f"scale={scale}, dw={dw}, dh={dh}")

    if gt_boxes is None:
        return image_padded

    else:
        if num_landmarks == 10:
            gt_boxes[:, [0, 2, 4, 6, 8, 10, 12]] = gt_boxes[:, [0, 2, 4, 6, 8, 10, 12]] * scale + dw
            gt_boxes[:, [1, 3, 5, 7, 9, 11, 13]] = gt_boxes[:, [1, 3, 5, 7, 9, 11, 13]] * scale + dh
            gt_boxes[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = np.clip(
                gt_boxes[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], 0., target_size[0]).astype(dtype=dtype)
        else:
            for box in gt_boxes:
                box[0] = np.clip(box[0] * scale + dw, 0, target_size[0]).astype(dtype=dtype)
                box[1] = np.clip(box[1] * scale + dh, 0, target_size[0]).astype(dtype=dtype)
                box[2] = np.clip(box[2] * scale + dw, 0, target_size[0]).astype(dtype=dtype)
                box[3] = np.clip(box[3] * scale + dh, 0, target_size[0]).astype(dtype=dtype)



                    # gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
                    # gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

        return image_padded, gt_boxes


def image_stats(image):
    """Compute the mean and standard deviation of each channel."""
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer(source, target):
    """Perform color transfer from the target to the source image."""
    # Convert the images from the BGR to Lab color space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # Subtract the means from the source image
    (l, a, b) = cv2.split(source)
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc

    # Scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # Add in the target mean
    l += lMeanTar
    a += aMeanTar
    b += bMeanTar

    # Clip the pixel intensities to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge the channels together and convert back to the BGR color space
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # Return the color transferred image
    return transfer


def scale_image(image, scale_factor):

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate center point
    center = (width // 2, height // 2)

    # Create scaling matrix
    scaling_matrix = np.array([[scale_factor, 0, (1 - scale_factor) * center[0]],
                               [0, scale_factor, (1 - scale_factor) * center[1]],
                               [0, 0, 1]])

    # Apply scaling matrix to the image with black border
    return cv2.warpAffine(image, scaling_matrix[:2, :], (width, height), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))


def align_crop_image(original_image, original_landmarks, landmarks_padding_ratio):
    """
    pads the image with half of max(image dimensions),
    rotates the image based on the face center (calculated from landmarks),
    and crops the face using the distance between landmarks from the padding ratio.
    Accepts 68 or 5 landmarks
    :param original_image:
    :param original_landmarks:
    :param landmarks_padding_ratio:
    :return: the center of the padded and rotated image is added to the returned rotated_landmarks and cropped_landmarks
    """
    # if isinstance(original_image, torch.Tensor):
    #     original_image


    secs = time.time()
    padded_image, padded_landmarks, pad_size = pad_image(original_image, original_landmarks)  # padding defaults to max(w, h) of image on each side; 4x original size
    # print("pad_image", time.time() - secs)
    if len(padded_image) == 0:
        raise ValueError("padded_image, this shit is empty")

    secs = time.time()
    # # print("rotate_image")
    rotated_image, rotated_landmarks, rotated_angle, qsize = rotate_image(
        image=padded_image,
        landmarks=padded_landmarks)
    # print("rotate_image", time.time() - secs)
    if rotated_image is None or len(rotated_image) == 0:
        raise ValueError("rotated_image, this shit is empty")

    # crop an image from the center point of the rotated image, size determined from landmarks and the landmarks_padding_ratio
    secs = time.time()
    cropped_image, cropped_landmarks, crop_box = center_crop(
        image=rotated_image,
        center=rotated_landmarks[-1],
        landmarks=rotated_landmarks,
        landmarks_padding_ratio=landmarks_padding_ratio,
        qsize=qsize
    )
    # print("center_crop", time.time() - secs)
    if len(cropped_image) == 0:
        raise ValueError("cropped_image, this shit is empty")

    # # center crop with size*2 then rotate to include every pixel; in theory, should be faster because of less calcs
    #

    cropped_params = {
        "original_landmarks": original_landmarks,
        "rotated_landmarks": rotated_landmarks,  # includes the rotated center (int); must use to paste back
        "rotated_angle": rotated_angle,
        "crop_box": crop_box,
        "cropped_landmarks": cropped_landmarks,
        "cropped_image_width": cropped_image.shape[1],
        "cropped_image_height": cropped_image.shape[0],
        "landmarks_padding_ratio": landmarks_padding_ratio,
    }
    return cropped_image, cropped_params


def paste_swapped_image(dst_image, swapped_image, seg_mask, aligned_cropped_params, rotate=True, resize=False, seamless_clone=True, blur_mask=True, face_scale=1):
    secs = time.time()
    dst_cropped_width = aligned_cropped_params["cropped_image_width"]
    cropped_landmarks = aligned_cropped_params["cropped_landmarks"]
    og_rotated_landmarks = aligned_cropped_params["rotated_landmarks"]
    rotated_angle = aligned_cropped_params["rotated_angle"]
    crop_box = aligned_cropped_params["crop_box"]

    # get the original height and width before padding
    h, w = dst_image.shape[:2]
    # pad the image
    dst_image, dst_landmarks, dst_pad_size = pad_image(dst_image, aligned_cropped_params["original_landmarks"])
    half_pad_size = int(dst_pad_size / 4)
    # print(f"pad image={time.time()-secs} secs")
    # rotate and resize the image
    secs = time.time()
    rotated_image, rotated_landmarks = resize_face_to_original_and_rotate(
        swapped_image,
        np.ones((*swapped_image.shape[:2], 1)) * 255 if seg_mask is None else seg_mask,  # use a plain mask if there's no seg mask
        original_size=(dst_cropped_width, dst_cropped_width),
        original_landmarks=cropped_landmarks,
        original_angle=rotated_angle,
        resize=resize)

    if not resize:
        rotated_blank_mask = rotated_landmarks
    # print(f"rotate image={time.time() - secs} secs")

    rotated_image, og_rotated_landmarks, pad_image_custom_size(rotated_image, og_rotated_landmarks, half_pad_size*2)

    # get the center that we cropped from for seamless close
    rotated_center = og_rotated_landmarks[-1]
    face_center = (int(rotated_center[0]), int(rotated_center[1]))

    # change image dtype for cv2 seamless clone
    image = rotated_image[..., :3].astype(np.uint8)
    mask = rotated_image[..., 3].astype(np.uint8)

    # scale image
    if face_scale != 1.0:
        image = scale_image(image, face_scale)
        mask = scale_image(mask, face_scale)

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    # blur the mask
    if blur_mask:
        # secs = time.time()
        # sigma = int(mask.shape[0] * 0.03) * 2 + 1
        # mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=21, sigmaY=21)
        # mask = gradient_blur_mask(mask)
    # # else:
    # #     mask = cv2.blur(mask, (16, 16))
    # else:
        mask = cv2.blur(mask, (15, 15))

    # make sure the mask has channels(3rd) dimension
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    # if not resize:
    #     mask[rotated_blank_mask != 255] = 0
    # do seamless clone
    secs = time.time()

    # array_to_img(mask).save(
    #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/seg_mask-{time.time()}.jpg",
    #     "JPEG")
    # array_to_img(seg_mask).save(
    #     f"C:/Users/teckt/PycharmProjects/iae_dfstudio/dfs_face_extractor/preds/seg_mask-{queue_index}-{image_index}-{face_index}.jpg",
    #     "JPEG")
    if seamless_clone:
        mask = np.ones_like(mask).astype(np.uint8)*255

        # make mask non-zero for seamless clone
        if not resize:
            mask = cv2.resize(mask, (dst_cropped_width, dst_cropped_width),
                       interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (dst_cropped_width, dst_cropped_width),
                       interpolation=cv2.INTER_CUBIC)
        # mask = np.clip(mask + 1, 0, 255).astype(np.uint8)

        # mask = np.ones_like(image).astype(np.uint8) * 255
        # mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=16, sigmaY=16)
        merged_image = cv2.seamlessClone(image[...,:3].astype(np.uint8), dst_image.astype(np.uint8), mask, face_center, cv2.NORMAL_CLONE)
        # dst_image[top_left_y:top_left_y + dst_cropped_width, top_left_x:top_left_x + dst_cropped_width] = image
        # merged_image = dst_image
    else:

        dtype = np.float32
        alpha = mask.astype(dtype)/255.
        # alpha blend zeroes like and merged image
        beta = (1.0 - alpha)

        crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box

        # only multiply the resized cropped portion equal to the original swapped face size (224x224) to save resources
        dst_image_cropped = dst_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
        if not resize:
            try:
                dst_image_cropped_resized = cv2.resize(dst_image_cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            except cv2.error:
                return np.clip(dst_image, 0, 255)
        else:
            dst_image_cropped_resized = dst_image_cropped

        for ch in range(3):
            # print(f"multiplying ch index {ch}")
            image_ch = np.expand_dims(image[..., ch], axis=-1).astype(dtype)
            dst_ch = np.expand_dims(dst_image_cropped_resized[..., ch], axis=-1).astype(dtype)
            # dst_ch = np.expand_dims(dst_image[top_left_y:top_left_y+dst_cropped_width, top_left_x:top_left_x+dst_cropped_width, ch], axis=-1).astype(dtype)
            # print("alpha", alpha.shape, "image", image_ch.shape, "dst", dst_ch.shape, "beta", beta.shape)

            image_alpha = cv2.multiply(image_ch, alpha)
            dst_beta = cv2.multiply(dst_ch, beta)
            merged_ch = cv2.add(image_alpha, dst_beta)

            dst_image_cropped_resized[..., ch] = merged_ch
            # dst_image[top_left_y:top_left_y+dst_cropped_width, top_left_x:top_left_x+dst_cropped_width, ch] = merged_ch

        # resize the merged cropped portion and paste it back to the original
        dst_image_cropped_resized = cv2.resize(dst_image_cropped_resized, (crop_x_max-crop_x_min, crop_y_max-crop_y_min), interpolation=cv2.INTER_LANCZOS4)
        try:
            dst_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = dst_image_cropped_resized
        except ValueError:
            pass

        # Blend the images using the alpha channel as the mask
        # merged_image = cv2.convertScaleAbs(dst_image * beta + alpha_zero[..., :3] * alpha)

        merged_image = np.clip(dst_image, 0, 255)
    # print(f"merge image={time.time() - secs} secs")

    # crop out the original image
    secs = time.time()

    merged_image = merged_image[half_pad_size:half_pad_size + h, half_pad_size:half_pad_size + w]
    # merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
    # print(f"final crop image={time.time() - secs} secs")

    return merged_image[..., :3]


def paste_image_with_mask(dst_image, image_to_paste, mask):
    h, w = dst_image.shape[:2]
    dtype = np.float32
    alpha = mask.astype(dtype) / 255.
    # alpha blend zeroes like and merged image
    beta = (1.0 - alpha)

    crop_x_min, crop_y_min, crop_x_max, crop_y_max = 0, 0, w, h

    for ch in range(3):
        # print(f"multiplying ch index {ch}")
        image_ch = np.expand_dims(image_to_paste[..., ch], axis=-1).astype(dtype)
        dst_ch = np.expand_dims(dst_image[..., ch], axis=-1).astype(dtype)
        # dst_ch = np.expand_dims(dst_image[top_left_y:top_left_y+dst_cropped_width, top_left_x:top_left_x+dst_cropped_width, ch], axis=-1).astype(dtype)
        # print("alpha", alpha.shape, "image", image_ch.shape, "dst", dst_ch.shape, "beta", beta.shape)

        image_alpha = cv2.multiply(image_ch, alpha)
        dst_beta = cv2.multiply(dst_ch, beta)
        merged_ch = cv2.add(image_alpha, dst_beta)

        dst_image[..., ch] = merged_ch
        # dst_image[top_left_y:top_left_y+dst_cropped_width, top_left_x:top_left_x+dst_cropped_width, ch] = merged_ch

    merged_image = np.clip(dst_image, 0, 255, dtype=np.uint8)
    return merged_image

def adjust_mask_for_image_black_areas_after_rotate(image, mask, order=0, size_factor=1.0):
    # create a mask where the black areas of the image is zeroes along all three channels and the rest are ones
    image_ones = np.all(image != 0, axis=-1, keepdims=True)
    # black_area = (np.zeros_like(image_ones) + image_ones) * 255
    black_area = (np.zeros_like(image_ones) + image_ones).astype(np.uint8)
    # print("black_area", black_area.shape)

    # erode/shrink the areas of the image with ones
    # eroded_black_area = pad_and_resize(black_area.astype(np.uint8),
    #                                    pad_size=int(black_area.shape[0] * pad_ratio) + 1)

    # erode_iterations = pad_ratio
    if order == 0:
        # at least 16 for blur

        kernel_size = int(black_area.shape[0] * 0.05 * size_factor)

        # Apply dilation to the image and erode again; fixes random dots transparent dots in face
        erosion_kernel_size = int(round(kernel_size * 0.25))
        dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        black_area = cv2.dilate(black_area.astype(np.uint8), dilation_kernel, iterations=1)
        eroded_black_area = cv2.erode(black_area.astype(np.uint8), erosion_kernel, iterations=1)

        eroded_black_area = np.expand_dims(eroded_black_area, axis=-1)
    else:
        # do a final erode that clips just enough to remove the black areas
        kernel_size = int(black_area.shape[0] * 0.02 * size_factor)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        eroded_black_area = cv2.erode(black_area.astype(np.uint8), kernel, iterations=1)
        # eroded_black_area = np.expand_dims(eroded_black_area / 255, axis=-1)
        eroded_black_area = np.expand_dims(eroded_black_area, axis=-1)

    mask = cv2.multiply(mask, eroded_black_area)

    return mask


def resize_face_to_original_and_rotate(face_image, face_mask, original_size, original_landmarks, original_angle, resize=True):
    # resize the mask to face image for concatenation
    h, w = face_image.shape[:2]
    if face_mask.shape[0] * face_mask.shape[1] > w * h:
        face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
    else:
        face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_CUBIC)
    # expand to 3 dims if a greyscale image's channel dimension was reduced
    if len(face_mask.shape) == 2:
        face_mask = np.expand_dims(face_mask, axis=-1)

    face_image_with_mask = np.concatenate((face_image, face_mask), axis=-1)

    center = (int(w / 2), int(h / 2))
    # rotate the image before resizing
    rotation_matrix = cv2.getRotationMatrix2D(center, -original_angle, 1)
    rotated_image = cv2.warpAffine(face_image_with_mask, rotation_matrix, face_image_with_mask.shape[:2],
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))

    blank_mask = np.ones_like(face_mask).astype(np.uint8) * 255
    rotated_blank_mask = cv2.warpAffine(blank_mask, rotation_matrix, face_image_with_mask.shape[:2],
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if not resize:
        return rotated_image, rotated_blank_mask

    if rotated_image.shape[0] * rotated_image.shape[1] > original_size[0] * original_size[1]:
        rotated_image = cv2.resize(rotated_image, original_size, interpolation=cv2.INTER_AREA)
    else:
        rotated_image = cv2.resize(rotated_image, original_size, interpolation=cv2.INTER_CUBIC)

    rotated_image = np.clip(rotated_image, 0, 255)

    # Convert the landmarks to homogeneous coordinates; add the center and +1 t0 homo
    landmarks_homo = np.concatenate((original_landmarks, np.ones((5+1, 1))), axis=1)

    # Apply the rotation transformation to the landmarks
    rotated_landmarks = np.dot(landmarks_homo, rotation_matrix.T)[:, :2]
    rotated_landmarks = np.concatenate([rotated_landmarks, [center]], axis=0)

    return rotated_image, rotated_landmarks


def center_crop(image, center, landmarks, landmarks_padding_ratio, qsize):
    # Calculate the bounding box from landmarks
    # x_min = np.min(landmarks[:, 0])
    # x_max = np.max(landmarks[:, 0])
    # y_min = np.min(landmarks[:, 1])
    # y_max = np.max(landmarks[:, 1])

    # height = width = qsize  # y_max - y_min

    # get max distance from bbox coords

    # crop_size = int(qsize * landmarks_padding_ratio)
    try:
        h=w=qsize
    except ValueError:
        h=w=min(image.shape[:2])

    crop_size = int(min(
        image.shape[0], image.shape[1],
        max(h, w) * landmarks_padding_ratio
    ))

    # if crop_size % 2 != 0:
    #     # add one to center and crop size to balance the offset
    #     crop_size = crop_size + 1
    #     landmarks[5][0] = landmarks[5][0] - 1
    #     landmarks[5][1] = landmarks[5][1] - 1

    half_crop_size = int(crop_size / 2)

    top_left_x = round(center[0]) - half_crop_size
    top_left_y = round(center[1]) - half_crop_size

    crop_box = [
        top_left_x,
        top_left_y,
        top_left_x + crop_size,
        top_left_y + crop_size
    ]

    # Perform the center crop
    cropped_image = image[top_left_y:top_left_y+crop_size, top_left_x:top_left_x+crop_size]

    if len(cropped_image) == 0:
        print(cropped_image)
        print(crop_box)
        cropped_image = np.ones_like(image, dtype='uint8') * 255

    # Adjust the landmark coordinates relative to the crop parameters
    adjusted_landmarks = []
    for landmark in landmarks:
        adjusted_x = landmark[0] - top_left_x
        adjusted_y = landmark[1] - top_left_y
        adjusted_landmarks.append([adjusted_x, adjusted_y])

    return cropped_image, adjusted_landmarks, crop_box


def get_face_center(landmarks):
    if len(landmarks) == 5:
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

    elif len(landmarks) == 68:
        nose_center_index = 30

        left_eye = landmarks[36]
        right_eye = landmarks[45]

        left_mouth = landmarks[48]
        right_mouth = landmarks[54]

    eyes_x = (right_eye[0] + left_eye[0]) / 2.
    eyes_y = (right_eye[1] + left_eye[1]) / 2.
    eyes_center = (eyes_x, eyes_y)

    mouth_x = (right_mouth[0] + left_mouth[0]) / 2.
    mouth_y = (right_mouth[1] + left_mouth[1]) / 2.
    mouth_center = (mouth_x, mouth_y)

    # face lib
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)
    left_mouth = np.array(left_mouth)
    right_mouth = np.array(right_mouth)

    eye_avg = (left_eye + right_eye) * 0.5
    eye_to_eye = right_eye - left_eye
    mouth_avg = (left_mouth + right_mouth) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Get the oriented crop rectangle
    # x: half width of the oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
    # norm with the hypotenuse: get the direction
    try:
        x /= np.hypot(*x)  # get the hypotenuse of a right triangle
    except:
        return 0,0,0,0
        x = 0
    rect_scale = 0.9  # TODO: you can edit it to get larger rect
    x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
    # y: half height of the oriented crop rectangle
    y = np.flipud(x) * [-1, 1]

    # c: center
    c = eye_avg + eye_to_mouth * 0.2
    # quad: (left_top, left_bottom, right_bottom, right_top)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    # qsize: side length of the square
    qsize = np.hypot(*x) * 2



    dy = eyes_center[1] - mouth_center[1]
    dx = eyes_center[0] - mouth_center[0]

    return c, dx, dy, qsize

    center_y = (eyes_center[1] + mouth_center[1]) / 2.
    center_x = (eyes_center[0] + mouth_center[0]) / 2.
    face_center = (int(round(center_x)), int(round(center_y)))
    return face_center, dx, dy


def rotate_image(image, landmarks):
    # print("pre_process_landmarks", landmarks)

    # Extract the coordinates of the eye and nose landmarks
    face_center, dx, dy, qsize = get_face_center(landmarks)
    if qsize == 0:
        return None, None, None, None


    # # switch
    # if left_eye[0] - right_eye[0] > 0:
    #     left_eye = landmarks[1]
    #     right_eye = landmarks[0]

    # nose_center = landmarks[2]

    # Calculate the angle between the line connecting the eye landmarks and the horizontal axis

    # center = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
    # angle = np.degrees(np.arctan2(dy, dx))

    angle = np.degrees(np.arctan2(dy, dx)) + 90
    # print("angle", angle, "from dy.dx.", dy, dx)

    # Rotate the image to align the face upright
    h, w, _ = image.shape
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)  # flags=cv2.INTER_LANCZOS4

    # Convert the landmarks to homogeneous coordinates +1 for the face center
    landmarks_homo = np.concatenate((np.concatenate([landmarks, [face_center]], axis=0), np.ones((len(landmarks)+1, 1))), axis=1)

    # Apply the rotation transformation to the landmarks
    landmarks_rotated = np.dot(landmarks_homo, rotation_matrix.T)[:, :2]  # Discard the homogeneous coordinate
    # print("landmarks_rotated", landmarks_rotated)

    return rotated_image, landmarks_rotated, angle, qsize


def pad_image(image, landmarks):
    h, w = image.shape[:2]
    max_side = max(h, w)
    pad_size = max_side
    # pad_size = max_side/2
    # make sure the pad size is divisible by 2
    pad_size = int(pad_size if pad_size % 2 == 0 else pad_size + 1)
    half_pad_size = int(pad_size / 4)
    # print("shape", image.shape, "max_side", max_side, "pad_size", pad_size)

    padded_black_background = np.zeros(shape=(max_side+(half_pad_size*2), max_side+(half_pad_size*2), image.shape[2]), dtype=np.uint8)
    # padded_black_background += 127
    # paste image to background
    # padded_image = array_to_img(padded_black_background).convert('RGBA')


    # padded_image.paste(image, ([half_pad_size, half_pad_size], [half_pad_size+w, half_pad_size+h]))
    # padded_image.paste(array_to_img(image).convert('RGBA'), (half_pad_size, half_pad_size))

    # Paste the image onto the background
    padded_black_background[half_pad_size:half_pad_size + h, half_pad_size:half_pad_size + w] = image

    # adjust landmarks by adding half pad size to each landmark
    padded_landmarks = []
    for (x, y) in landmarks:
        padded_landmarks.append([x+half_pad_size, y+half_pad_size])

    # # convert back to array
    # padded_image = img_to_array(padded_image)
    return padded_black_background, padded_landmarks, pad_size


def pad_image_custom_size(image, landmarks, pad_size):
    h, w = image.shape[:2]
    max_side = max(h, w)

    # make sure the pad size is divisible by 2
    pad_size = int(pad_size if pad_size % 2 == 0 else pad_size + 1)
    half_pad_size = int(pad_size / 2)

    padded_black_background = np.zeros(shape=(max_side+(half_pad_size*2), max_side+(half_pad_size*2), image.shape[2]), dtype=np.uint8)

    # Paste the image onto the background
    padded_black_background[half_pad_size:half_pad_size + h, half_pad_size:half_pad_size + w] = image

    # adjust landmarks by adding half pad size to each landmark
    padded_landmarks = []
    for (x, y) in landmarks:
        padded_landmarks.append([x+half_pad_size, y+half_pad_size])

    return padded_black_background, padded_landmarks, pad_size

def gradient_blur_mask(mask):
    padding = mask.shape[0]//2
    padded_mask = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    image_size = mask.shape[0]

    # # Define the kernel (structuring element) for dilation
    # kernel_size = 7
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #
    # # Measure the execution time
    # start_time = time.time()
    # # Perform dilation on the seg mask
    # dilated_mask = cv2.dilate(padded_mask, kernel, iterations=2)
    #
    # end_time = time.time()
    # execution_time = end_time - start_time
    #
    # # print(f"dilated_mask Execution time: {execution_time:.6f} seconds")

    # Measure the execution time
    start_time = time.time()
    # Apply Gaussian blur to the padded mask following the same value used for kernel size in erosion

    sigma = int(image_size * 0.01) * 2 + 1
    blurred_mask = cv2.GaussianBlur(padded_mask, (5, 5), sigmaX=sigma, sigmaY=sigma)

    # Crop the blurred mask to remove the padding
    # blurred_mask = blurred_mask[padding:-padding, padding:-padding]
    sigma = int(sigma * 1.25)
    blurred_mask = blurred_mask[padding - sigma:-(padding - sigma), padding - sigma:-(padding - sigma)]
    blurred_mask = cv2.resize(blurred_mask, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return blurred_mask

    end_time = time.time()
    execution_time = end_time - start_time

    # print(f"blurred_mask Execution time: {execution_time:.6f} seconds")

    # Image dimensions

    gradient_width = max(int(image_size*0.03), 5)

    gradient_width_h = max(int(image_size * 0.03), 5)
    # Create the initial grayscale image with ones
    initial_image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    initial_image[:gradient_width, :] = 0
    initial_image[-gradient_width:, :] = 0
    initial_image[:, :gradient_width_h] = 0
    initial_image[:, -gradient_width_h:] = 0
    initial_image = cv2.GaussianBlur(initial_image, (5, 5), sigmaX=gradient_width, sigmaY=gradient_width_h)
    initial_image = np.clip(initial_image / 255., 0., 1.)

    blurred_mask = np.clip(blurred_mask * initial_image, 0, 255).astype(np.uint8)

    # Create the initial grayscale image with ones
    initial_image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Create a linear gradient from 0 to 255
    gradient = np.linspace(0, 255, gradient_width, dtype=np.uint8)
    gradient_h = np.linspace(0, 255, gradient_width_h, dtype=np.uint8)

    # Set the edges of the image with the gradient
    initial_image[:gradient_width, :] = gradient.reshape(gradient_width, 1)
    initial_image[-gradient_width:, :] = gradient[::-1].reshape(gradient_width, 1)
    initial_image[:, :gradient_width_h] = gradient_h
    initial_image[:, -gradient_width_h:] = gradient_h[::-1]

    # gradient = 255
    # for i in range(gradient_width):
    #     squared_i = (gradient_width-i)**2
    #     initial_image[:i, :] = (gradient/squared_i)
    #     initial_image[-i:, :] = (gradient/squared_i)
    #     initial_image[:, :i] = (gradient/squared_i)
    #     initial_image[:, -i:] = (gradient/squared_i)
    initial_image = np.clip(initial_image / 255., 0., 1.)

    blurred_mask = np.clip(blurred_mask * initial_image * 1.0, 0, 255).astype(np.uint8)

    # try:
    #     blurred_mask.save("C:/Users/teckt/PycharmProjects/iae_dfstudio/models/iae/tflite_v2/test/seg_mask.jpg")
    # except OSError:
    #     pass

    return blurred_mask