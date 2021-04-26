import random
from Maths.Functions import measureTime

@measureTime
def getPoint(minX, maxX, minY, maxY):
    x = random.randrange(minX, maxX)
    y = random.randrange(minY, maxY)

    return x, y

