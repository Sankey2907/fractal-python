import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu

def box_count(image, box_size):
    # Calculate the number of boxes in the x and y directions
    num_boxes_x = np.ceil(image.shape[1] / box_size).astype(int)
    num_boxes_y = np.ceil(image.shape[0] / box_size).astype(int)

    # Create a 2D array of box indices
    row_indices = np.arange(num_boxes_y) * box_size
    col_indices = np.arange(num_boxes_x) * box_size
    box_indices = np.ix_(row_indices, col_indices)

    # Count the number of boxes that intersect with the foreground
    # that means from which the boxes are clearly visible 
    boxes = image[box_indices]
    box_count = np.count_nonzero(boxes)

    return box_count

# Load the image and convert it to grayscale
#add link of img
image = io.imread('C:\\Users\\Lenovo\\Downloads\\f.png', as_gray=True)

# Apply Otsu's thresholding to segment the image
# otsu thresholding use to provide automatic image thresholding
thresh = threshold_otsu(image)
binary = image > thresh

# Define the range of box sizes to use
min_box_size = 2
max_box_size = min(image.shape[:2])
box_sizes = 2 ** np.arange(np.ceil(np.log2(min_box_size)), 
                            np.floor(np.log2(max_box_size)), dtype=int)

# Count the number of boxes that intersect with the foreground for each box size
box_counts = []
for size in box_sizes:
    box_counts.append(box_count(binary, size))

# Fit a straight line to the log-log plot of box count vs box size
# this process provide the graph of  image 
coeffs = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)
fractal_dimension = -coeffs[0]

# Print the fractal dimension
print(f'The fractal dimension is {fractal_dimension:}')

# Plot the log-log plot of box count vs box size
plt.figure()
plt.loglog(box_sizes, box_counts, 'bo')
plt.loglog(box_sizes, np.exp(np.polyval(coeffs, np.log(box_sizes))), 'r-')
plt.xlabel('Box size')
plt.ylabel('Box count')
plt.title(f'Fractal dimension: {fractal_dimension:}')
plt.show()


