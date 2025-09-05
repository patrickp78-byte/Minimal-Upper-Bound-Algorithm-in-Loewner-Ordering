import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from outer_ellipsoid import outer_ellipsoid_fit, outer_ellipsoid_fit_2

matplotlib.use("TkAgg")

# Generate points (2D now)
pts = np.random.rand(100, 2)

# Compute ellipsoids
A1, c1 = outer_ellipsoid_fit(pts)     # original
A2, c2 = outer_ellipsoid_fit_2(pts)   # Minimal upper bound

# Function to create ellipse boundary from quadratic form
def create_ellipse(A, center, num_points=200):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 1e-8, None)
    axes = 1 / np.sqrt(eigvals)

    t = np.linspace(0, 2 * np.pi, num_points)
    circle = np.stack((np.cos(t), np.sin(t)))
    ellipse = eigvecs @ np.diag(axes) @ circle
    ellipse = ellipse + center[:, np.newaxis]
    return ellipse

# Create ellipses
ellipse1 = create_ellipse(A1, c1)
ellipse2 = create_ellipse(A2, c2)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(pts[:, 0], pts[:, 1], c='blue', label='Data Points', s=10)

ax.plot(ellipse1[0], ellipse1[1], color='green', label='Loewner Outer Ellipse')
ax.plot(ellipse2[0], ellipse2[1], color='red', label='Minimal Upper Bound Outer Ellipse')

# Compute areas
area1 = np.pi / np.sqrt(np.linalg.det(A1))
area2 = np.pi / np.sqrt(np.linalg.det(A2))
print(f"Loewner ellipse area: {area1:.4f}")
print(f"Minimal upper bound ellipse area: {area2:.4f}")
print(f"Ratio (MUB / Loewner): {area2/area1:.4f}")
print(f"Difference (MUB - Loewner): {area2-area1:.4f}")

ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.tight_layout()
plt.show()
