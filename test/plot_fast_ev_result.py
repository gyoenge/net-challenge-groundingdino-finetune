import matplotlib.pyplot as plt

# speed data
resolutions = ["UHD 3840x2160", "QHD 2560x1440", "FHD 1920x1080", "HD 1280x720", "SVGA 800x600", "VGA 640x480", "QVGA 320x240"]
data = [
    [324.53, 325.24, 324.92, 326.35, 321.19, 322.84, 323.39, 324.97, 323.68, 326.32],
    [286.81, 287.02, 288.93, 290.19, 283.79, 285.89, 287.5, 288.16, 287.34, 287.24],
    [278.29, 278, 274.38, 275.63, 273.89, 271.9, 273.9, 278.21, 273.37, 271.53],
    [271.43, 265.16, 265.76, 268.81, 264.86, 269.69, 262.39, 259.45, 257.45, 261.17],
    [260.72, 257.23, 256.94, 262.99, 262.6, 263.35, 261.95, 257.04, 256.25, 259.58],
    [256.92, 256.79, 258.6, 260.13, 261.43, 259.45, 252.91, 256.13, 254.3, 259.45],
    [258.08, 256.53, 256.82, 258.72, 260.19, 259.03, 252.41, 253.87, 254.3, 258.72]
]


# plot graph 
for i, resolution in enumerate(resolutions):
    plt.plot(data[i], label=resolution)

plt.title('Grounding DINO Time for Different Resolutions')
plt.xlabel('Experiment Number')
plt.ylabel('Time (s)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()