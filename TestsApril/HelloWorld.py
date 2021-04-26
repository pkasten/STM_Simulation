import numpy as np
from Configuration import ConfigManager

x = np.linspace(1, 10, 3)


ConfigManager.set_threads(16)
print(ConfigManager.get_threads())

