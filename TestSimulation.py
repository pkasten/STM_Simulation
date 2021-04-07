from Data import Simulation as sim
from Data import DataFrame as frame
from Visualization import Images


def test():
    img = Images.Images()
    number_of_points = 10

    data = frame.DataFrame()
    for i in range(number_of_points):
        data.addPoint(sim.getPoint(0, img.getWidth(), 0, img.getHeight()))

    img.createImage(data)

    # print(data)


if __name__ == "__main__":
    test()
