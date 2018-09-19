import mayavi
from mayavi import mlab
import numpy as np

mlab.clf()  # Clear the figure
t = np.linspace(0, 20, 200)
mlab.plot3d(np.sin(t), np.cos(t), 0.1*t, t)

mlab.show()