import matplotlib.pyplot as plt


def cv2_imshow(image, title=''):
    plt.imshow(image[:, :, [2, 1, 0]])
    plt.title(title)
    plt.show()