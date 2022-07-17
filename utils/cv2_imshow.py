import matplotlib.pyplot as plt


def cv2_imshow(image, title=''):
    plt.imshow(image)
    plt.title(title)
    plt.show()