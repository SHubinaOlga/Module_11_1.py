import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image, ImageFilter

"""
Использование модуля matplotlib. 
Визуализация данных с помощью инструментом из библиотеки.
"""
SUCCESS_SYMBOLS = [
    TextPath((0, 0), "☹"),
    TextPath((0, 0), "😒"),
    TextPath((0, 0), "☺"),
]
N = 25
np.random.seed(42)
skills = np.random.uniform(5, 80, size=N) * 0.1 + 5
takeoff_angles = np.random.normal(0, 90, N)
thrusts = np.random.uniform(size=N)
successful = np.random.randint(0, 3, size=N)
positions = np.random.normal(size=(N, 2)) * 5
data = zip(skills, takeoff_angles, thrusts, successful, positions)

cmap = plt.colormaps["plasma"]
fig, ax = plt.subplots()
fig.suptitle("Throwing success", size=14)
for skill, takeoff, thrust, mood, pos in data:
    t = Affine2D().scale(skill).rotate_deg(takeoff)
    m = MarkerStyle(SUCCESS_SYMBOLS[mood], transform=t)
    ax.plot(pos[0], pos[1], marker=m, color=cmap(thrust))
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
             ax=ax, label="Normalized Thrust [a.u.]")
ax.set_xlabel("X position [m]")
ax.set_ylabel("Y position [m]")

plt.show()

""" 
Использование модуля NumPy.
Сложение поэлементно.
"""
x = np.array([[1,10],[3,2]], dtype=np.float64)
y = np.array([[7,6],[11,8]], dtype=np.float64)
arr = np.array([3, 2])

print(x + y)
print()
print(np.add(x, y))
print('С числом')
print(x + 1)
print('C массивом другой размерности')
print(x + arr)

"""
Использование модуля pillow.
Наложение картинки на картинку.
"""
filename_w = 'кот.jpg'
with Image.open(filename_w) as img_w:
     img_w.load()

img_w = img_w.crop((56, 0, 564, 852))
img_w.show()

img_w_gray = img_w.convert("L")
img_w_gray.show()
threshold = 100
img_w_threshold = img_w_gray.point(
     lambda x: 255 if x > threshold else 0)

img_w_threshold.show()

red, green, blue = img_w.split()
red.show()
green.show()
blue.show()

threshold_1 = 57
img_w_threshold_1 = blue.point(lambda x: 255 if x > threshold_1 else 0)
img_w_threshold = img_w_threshold.convert("1")
img_w_threshold_1.show()

def erode(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(3))
    return image
def dilate(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(3))
    return image

step_1 = erode(33, img_w_threshold)
step_1.show()

step_2 = dilate(95, step_1)
step_2.show()

w_mask = erode(52, step_2)
w_mask.show()

w_mask = w_mask.convert("L")
w_mask = w_mask.filter(ImageFilter.BoxBlur(30))
w_mask.show()

blank = img_w.point(lambda _: 0)
w_segmented = Image.composite(img_w, blank, w_mask)
w_segmented.show()

filename_1 = "небо.jpg"
with Image.open(filename_1) as img_1:
    img_1.load()

img_1.paste(
    img_w.resize((img_w.width // 1, img_w.height // 1)),
    (1900, 1800),
    w_mask.resize((w_mask.width // 1, w_mask.height // 1)), )

img_1.show()
