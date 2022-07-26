import matplotlib.pyplot as plt


def cv2_imshow(image, title=''):
    shape_len = len(image.shape)
    if shape_len == 3:
        h, w, c = image.shape
    else:
        c = 0
    if c == 3:
        plt.imshow(image[:, :, [2, 1, 0]])
    else:
        plt.imshow(image)
    plt.title(title)
    plt.show()


