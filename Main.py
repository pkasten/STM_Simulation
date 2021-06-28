from Molecule import Molecule
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math, time
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import matplotlib.pyplot as plt

"""
Main class used to interact with the program. Provides basic methods
Important are act(), GenExec and execNthreads()
"""

def test_frame(data_frame):
    """
    Adds Particles at steadily increasing angles line by line to the frame
    Useful to test if angles are defined correctly

    :param data_frame: The frame where particles should be added
    :return: None
    """

    maxn = len(range(50, cfg.get_width(), 100)) * len(range(50, cfg.get_height(), 100))
    th = 0
    dth = 2 * math.pi / maxn
    for y in range(50, cfg.get_height(), 100):
        for x in range(50, cfg.get_width(), 100):
            data_frame.addParticle(Particle(x, y, th))
            th += dth


def measure_speed():
    """
    Generates and saves images for increasing numbers of particles.
    Useful to measure correlation between number of particles and computational time
    Plots time over n

    :return: None
    """

    x = []
    y = []
    for i in range(20):
        start = time.perf_counter()
        dat_frame = DataFrame(FilenameGenerator())
        dat_frame.addParticles(amount=i, overlapping=False)
        dat_frame.get_Image()
        dat_frame.save()
        x.append(i)
        y.append(time.perf_counter() - start)

    plt.plot(x, y)
    plt.show()


def generate(fn_gen):
    """
    Creates an image w/ ordered particles

    :param fn_gen: Filename generator instance
    :return: None
    """
    dat_frame = DataFrame(fn_gen)
    dat_frame.add_Ordered()
    dat_frame.get_Image()
    dat_frame.save()


def multi_test():
    """
    Creates as many Generator instances as specified in Configuration-threds
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator()

    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()


class Generator(Process):
    """
    Multiprocessing process to execute generate()
    """
    def __init__(self, fn_gen):
        super().__init__()
        self.fn_gen = fn_gen

    def run(self):
        generate(self.fn_gen)




def multi_test_fn(t, fgen):
    """
    Runs instances of Generator and kills them after time t
    :param t: time in seconds to wait
    :param fgen: Filename Generator instance
    :return: None
    """
    fn_generator = fgen
    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()

    time.sleep(t)
    for gen in gens:
        gen.kill()





class Gen1(Process):
    """
    Generator that runs generate multiple times
    """
    def __init__(self, fn, n):
        """

        :param fn: Filename Generator istance
        :param n: number of executions
        """
        super().__init__()
        self.fn = fn
        self.n = n

    def run(self) -> None:
        for i in range(self.n):
            generate(self.fn)


def every_thread_n(n):
    """
    Lets every thread generate n Images
    :param n: Number of images
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        ts.append(Gen1(fn_gen, n))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return True


@DeprecationWarning
def test_Length_and_Sigma():
    """
    DEPRECATED.  Used to generate images with different parameters for angle correlation
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator()
    l = 0.001
    inc = 0.001
    while l < 3:
        cfg.set_angle_char_len(l)
        testStdderiv(fn_generator, l)
        if inc > 1:
            dinc = 0.2
        elif inc > 0.1:
            dinc = 0.01
        else:
            dinc = 0.002
        inc += dinc
        l += inc


class Gen2(Process):
    """
    Generator used for ordered adding at specific angles
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator instance
        :param a: angle ordered lattice is turned by
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        dat_frame.add_Ordered(Molecule, theta=self.a)
        dat_frame.get_Image()
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))


def every_angle():
    """
    Runs Gen2-Instances for angles from 0 to 360 degree
    :return: True to keep parent method alive
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        a = ((360 / cfg.get_threads()) * i) * np.pi / 180
        ts.append(Gen2(fn_gen, a))
    for t in ts:
        t.start()
        time.sleep(2)
    for t in ts:
        t.join()
    return True


class Gen3(Process):
    """
    Generator used to test Double-Tip method
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator instance
        :param a: angle in which direction the double tipping effect should happen
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        dat_frame.add_Ordered(theta=0)
        dat_frame.get_Image(ang=self.a)
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))




@DeprecationWarning
def testStdderiv(fn_gen, l):
    """
    DEPRECATED. Used to test different standard derivations for angle correlation
    :param fn_gen: Filename Generator instance
    :param l: Angle char, length currently used
    :return: None
    """
    fn = fn_gen
    s = 0.005
    inc = 0.01
    while s < 100:
        cfg.set_angle_stdderiv(s)
        print("Sigma = {}, l={} with Index {}".format(s, l, fn.generateIndex()))
        multi_test_fn(8, fn)
        if inc >= 2:
            dinc = 30
        elif inc >= 10:
            dinc = 6
        elif inc > 1:
            dinc = 0.1
        elif inc > 0.1:
            dinc = 0.2
        else:
            dinc = 0.02
        inc += dinc
        s += inc


def test_finv_acc():
    """
    Testing method to measure computational performance of Function.finv()
    Plots Time over accuracy
    :return: None
    """

    times = []
    accs = []
    for i in range(1, 1000, 5):
        start = time.perf_counter()
        f = lambda x: np.exp(x)
        finv = get_invers_function(f, 0, 400, i)

        xs = [i for i in range(400)]
        ys = [finv(i) for i in range(400)]
        times.append(time.perf_counter() - start)
        accs.append(i)

    plt.plot(accs, times)
    plt.title("Computing time over Accuracy")
    plt.show()


def test_finv_len():
    """
    Testing method to measure computational performance of Function.finv()
    Plots Time over number of points
    :return: None
    """

    times = []
    lens = []
    for i in range(1, 100, 5):
        start = time.perf_counter()
        f = lambda x: np.exp(x)
        finv = get_invers_function(f, 0, i)

        xs = [i for i in range(400)]
        ys = [finv(i) for i in range(400)]
        times.append(time.perf_counter() - start)
        lens.append(i)

    plt.plot(lens, times)
    plt.title("Computing time over Points")
    plt.show()


class Gen4(Process):
    """
    Generator for adding a single particle in the middle of the image rotated by given angle
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator Instance
        :param a: Angle
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        p = Particle(cfg.get_width() / 2, cfg.get_height() / 2, self.a)
        dat_frame.addParticle(p)
        dat_frame.get_Image()
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))


def every_angle2():
    """
    Invokes Gen4
    :return:
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        a = ((180 / cfg.get_threads()) * i) * np.pi / 180
        ts.append(Gen4(fn_gen, a))
    for t in ts:
        t.start()
        time.sleep(0.5)
    for t in ts:
        t.join()
    return True


"""
Add Code here
"""
thrds = cfg.get_threads()
recursions = 5


def act(dat):
    """
    Core method. This code will be run by each thread on the given DataFrame

    :param dat: Data frame opertaions should be done on
    :return:
    """
    #pos1 = np.array([Distance(True, 20), cfg.get_height() / 4])
    #pos2 = np.array([Distance(True, 50), cfg.get_height() / 4])
    #pos3 = np.array([Distance(True, 80), cfg.get_height() / 4])

    #m1 = Molecule(pos1, theta=0, molecule_ph_groups=3)
    #m2 = Molecule(pos2, theta=0, molecule_ph_groups=4)
    #m3 = Molecule(pos3, theta=0, molecule_ph_groups=5)

    #pos10 = np.array([Distance(True, 20), 6 * cfg.get_height() / 8])
    #pos20 = np.array([Distance(True, 50), 6 * cfg.get_height() / 8])
    #pos30 = np.array([Distance(True, 80), 6 * cfg.get_height() / 8])

    #m10 = Molecule(pos10, theta=0, molecule_ph_groups=3, style="Complex")
    #m20 = Molecule(pos20, theta=0, molecule_ph_groups=4, style="Complex")
    #m30 = Molecule(pos30, theta=0, molecule_ph_groups=5, style="Complex")

    #dat.addObject(m10)
    #dat.addObject(m1)
    #dat.addObject(m20)
    #p = Particle(cfg.get_width()/2, cfg.get_height()/2, 30)
    #mat1 = p.efficient_Matrix()[0]
    #for i in range(np.shape(mat1)[0]):
    #    for j in range(np.shape(mat1)[1]):
    #        if i < 7 or np.shape(mat1)[0] - i < 7 or j < 7 or np.shape(mat1)[1] - j < 7:
    #            mat1[i,j] = 200

    #mat2 = Functions.turnMatrix(mat1, np.pi/4)[0]

    #mat1 = Functions.turn_matplotlib(mat1)
    #mat2 = Functions.turn_matplotlib(mat2)
    #plt.subplot(121)
    #plt.title("Original Visualization Matrix")
    ##plt.imshow(mat1)
    #plt.subplot(122)
    #plt.title("Matrix turned by 45°")
    #plt.imshow(mat2)
    #plt.show()


    #dat.addObject(m2)
    #dat.addObject(m30)
    #dat.addObject(m3)

    #dat.addParticles(amount=1)

    #dat.add_Ordered(ph_grps=4, style="Simple")

    #dat.get_Image()
    #dat.save()
    #dat.addParticles(amount=20)
    dat.add_Ordered()
    #dat.add_Ordered()
    dat.get_Image()
    dat.save()


class GenExc(Process):
    """
    Core Generator. Code inside act() will be called amnt time by every thread
    """
    def __init__(self, fn, amnt):
        """

        :param fn: Filename Generator instance
        :param amnt: Number of time the code should be executed
        """
        super().__init__()
        self.fn = fn
        self.am = amnt

    def run(self) -> None:
        for i in range(self.am):
            dat = DataFrame(self.fn)
            act(dat)


def execNthreads(n, amnt=1):
    """
    Core method. Generates and runs n GenExcs
    :param n: Number of processes to start in parallel
    :param amnt: number of times each process should do the task in act()
    :return:
    """

    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(n):
        ts.append(GenExc(fn_gen, amnt))
    for t in ts:
        t.start()
        time.sleep(0.1)
    for t in ts:
        t.join()
    return True


if __name__ == "__main__":
    clearLog()

    execNthreads(thrds, recursions)

    # lo = Lock()
    # start = time.perf_counter()
    # fn = FilenameGenerator()
    #
    # every_angle2()
    # dat = DataFrame(fn)
    # p = Particle(cfg.get_width()/2, cfg.get_height()/2, 0)
    # dat.addParticle(p)
    # dat.get_Image()
    # dat.save()
    # dat = DataFrame(fn)
    # dat.add_Ordered()
    # dat.get_Image()
    # dat.save()
    # every_angle2()
    # mat = 200 * np.ones((200, 200))
    # for i in range(np.shape(mat)[0]):
    #    for j in range(np.shape(mat)[1]):
    #        col = 0
    #        if i > np.shape(mat)[0]/2:
    #            col += 75
    ##        else:
    #            col += 25

    #        if j > np.shape(mat)[1]/2:
    #            col += 100
    #        else:
    #            col += 0
    #        mat[i, j] = col
    # every_thread_n(10)
    # plt.imshow(mat)

    # plt.show()
    # for i in range(0, 360, 10):
    #    mat2, a, b = turn_matrix2(mat, i * 3.14159/180)
    #    plt.imshow(mat2)
    #    plt.show()

    # every_thread_n(1)
    # multi_test(1)
    # m = Molecule()
    # Tests_Gitterpot().test()

    # test_finv_acc()

    # for i in range(1):
    #    dat = DataFrame(fn)
    #    dat.addObjects(Molecule)
    #    #dat.add_Dust(DustParticle(np.array([200, 200]), color=i))
    #    dat.get_Image()
    #    dat.save()

    # start = time.perf_counter()
    # f = lambda x: x*np.sin(x/100) + 20
    # finv = get_invers_function(f)
    # xs = np.linspace(0, 400, 200)
    # ys = [finv(xy) for xy in xs]
    # plt.xkcd()
    # plt.plot(xs, ys)
    # every_angle()
    # p#lt.show()
    # print("Dur: {:.2f}s".format(time.perf_counter() - start))
    # times = []
    # accs = []
    # f#or i in range(1000):
    #    start = time.perf_counter()
    #    f = lambda x:np.exp(x)
    #    finv = get_invers_function(f, 0, 400, i)##

    #    xs = [i for i in range(400)]
    #   ys = [finv(i) for i in range(400)]
    #  times.append(time.perf_counter() - start)
    # accs.append(i)

    # plt.plot(accs, times)
    # plt.title("Computing time over Accuracy")
    # plt.show()
    # every_angle()
    # print(every_thread_n(10))
    # dat = DataFrame(fn)
    # dat.add_Ordered()
    # dat.get_Image()
    # dat.save()

    # dat = DataFrame(fn)
    # p = Particle(Distance(True, 35), Distance(True, 35), 0)
    # dat.addParticle(p)
    # dat.get_Image()
    # dat.save()
    # every_thread_n(5)
    # if every_thread_n(10):
    #    exit()
    #         time.sleep(5)

    # img = MyImage()
    # img.rgb_map_test()
    # for i in range(0, 360, 10):
    #    dat = DataFrame(fn)
    #    dat.add_Ordered(Molecule, theta=np.pi * i / 180)
    #    dat.get_Image()
    #    dat.save()
    ##    print("Dur: {:.2f}s".format(time.perf_counter() - start))
    #   start = time.perf_counter()
    ##    break
    # for i in range(5):
    #    dat = DataFrame(fn)
    #    dat.add_Ordered()
    #    dat.get_Image()
    #    dat.save()

    # DustParticle.test()

    # dat = DataFrame(fn)
    # for n in range(0,8, 3):
    #    dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200 + 100*n)]), n*0.785))
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200 + 100)]), 0))
    # dat.addObject(Particle(Distance(False, 200), Distance(False, 500), 0.9))
    # dat.addObjects(Molecule, amount=1)
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200)]), 1*0.785))
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 240)]), 2*0.785))
    # for i in range(30):
    #    print("Iteration I={}".format(i))
    #    dat.addObjects(Molecule, amount=10)

    # dat.objects[0].drag(10, 0)

    # start = time.perf_counter()
    # print(dat.has_overlaps())
    # print("Has Overlaps: " + str(time.perf_counter() - start))
    # start = time.perf_counter()
    # dat.get_Image()
    # print("Get Image: " + str(time.perf_counter() - start))
    # dat.save()

    # filename = "pySPM_Tests/HOPG-gxy1z1-p2020.sxm"
    # filename = "Test_SXM_File.sxm"
    # with open("SXM_ino.txt", "w") as file:
    #   file.write(My_SXM.get_informations(filename))
    # generate(fn)
    # data = np.random.random((256, 256))
    # My_SXM.write_sxm("Test2.sxm", data)
    # My_SXM.write_header("Test4.sxm")
    # My_SXM.write_image("Test4.sxm", data)
    # My_SXM.show_data("Test2.sxm")
    # generate(fn)
    # My_SXM.write_header(os.path.join("pySPM_Tests", "test_header_writer.sxm"))
    # print(My_SXM.get_informations("pySPM_Tests/test_header_writer.sxm"))
    # plt.imshow(My_SXM.get_data_test("pySPM_Tests/HOPG-gxy1z1-p2020.sxm"))
    # plt.show()
    # SXM_info.adjust_to_image(data, "Test3.sxm")
    # My_SXM.write_header("Test3.sxm")
    # My_SXM.write_image("Test3.sxm", data)
    # My_SXM.show_data("Test3.sxm")
    # generate(fn)
    # data = np.random.random((300, 300))
    # My_SXM.write_sxm("Test4.sxm", data)
    # def deg(x):
    #    return x * np.pi / 180

    # for theta in range(44, 45, 1):
    # multi_test(180)
    # dat_frame = DataFrame(fn)
    # for i in range(100):
    #    generate(fn)
    # multi_test(300)
    #    print(theta, fn.generateIndex())
    #   a = Particle(200, 200, deg(0))
    #   b = Particle(220, 220, deg(330))
    #   dat_frame.addParticle(a)
    #   dat_frame.addParticle(b)
    #   print("A overlaps b: {}".format(a.true_overlap(b)))
    # print("B overlaps a: {}".format(b.true_overlap(a)))
    # print("DF has overlaps: {}".format(dat_frame.has_overlaps()))
    #   dat_frame.get_Image()
    #   dat_frame.save()
    # every_thread_one()
    # generate(fn)
    # for i in range(360):
    #    dat = DataFrame(fn)
    #    dat.addParticle(Particle(300, 300, deg(i)))
    #    dat.calc_potential_map()
    #    img = MyImage()
    #    mat = np.zeros((600, 600))
    #    for i in range(600):
    #        for j in range(600):
    #            mat[i, j] = 125 + 125 * dat.potential_map[i, j]
    #    img.addMatrix(mat)
    #    img.updateImage()
    #    img.saveImage(fn.generate_Tuple()[0])

    # plt.imshow(dat.potential_map.transpose())
    # plt.savefig()
    #    dat.get_Image()
    #    dat.save()

    # dat = DataFrame(fn)
    # for i in range(60, cfg.get_width() - 60, 60):
    #    for j in range(60, cfg.get_height() - 60, 60):
    #        dat.add_at_optimum_energy_new(100 + 400 * random.random(), 100 + 400 * random.random(), 6.28 * random.random())
    #        dat.get_Image()
    #        dat.save()
    ##dat.add_ALL_at_optimum_energy_new(10)
    # dat.get_Image()
    # dat.save()

    # dat_frame.addParticle(Particle(100, 100, 0.5))
    # for i in range(5):
    #   a = Particle(200, 200, deg(0))
    #   b = Particle(220, 220, deg(330))
    #   dat_frame.addParticle(a)
    #   dat_frame.addParticle(b)
    #   print("A overlaps b: {}".format(a.true_overlap(b)))
    # print("B overlaps a: {}".format(b.true_overlap(a)))
    # print("DF has overlaps: {}".format(dat_frame.has_overlaps()))
    #   dat_frame.get_Image()
    #   dat_frame.save()
    # every_thread_one()
    # multi_test(1000000)
    # dat_frame.addParticle(Particle(100, 100, 0.5))
    # for i in range(5):
    #    dat_frame.add_at_optimum_energy_new(100 + 400 * random.random(), 100 + 400 * random.random(), 2* np.pi * random.random())
    # dat_frame.add_at_optimum_energy_new(200, 260, deg(180))
    # print(4)
    # dat_frame.add_at_optimum_energy([200, 300, deg(180)])
    # dat_frame.add_at_optimum_energy([250, 240, deg(180)])
    # dat_frame.get_Image()
    # dat_frame.save()
    # generate(fn)
    # for i in range(10):
    #    dat = DataFrame(fn)
    #    dat.addParticles(amount=10)
    #    dat.get_Image()
    #    dat.save()
    # dat_frame.potential_map = dat_frame.calc_potential_map()
    # for i in range(20):
    #    print(i)
    #    dat_frame.add_at_optimum_energy()
    #    dat_frame.potential_map = dat_frame.calc_potential_map()
    #    dat_frame.get_Image()
    #    dat_frame.save(sxm=False, data=False)
    # dat_frame.addParticle(b)
    # dat_frame.get_Image()
    # dat_frame.save()
    # print(dat_frame.has_overlaps())
    # print(a.true_overlap(b))
    # print(1)
    # generate(fn)
    # print(2)
    # generate(fn)
    # print(3)
    # for i in range(10):
    #    generate(fn)
    # multi_test(30)
    # generate(fn)
    # My_SXM.show_data("sxm/Image1.sxm")
    # SXM_info.adjust_to_image(data, "Test4.sxm")
    # My_SXM.write_header("Test4.sxm")
    # My_SXM.write_image("Test4.sxm", data)
    # My_SXM.show_data("Test4.sxm")
    # My_SXM.show_data("Test4.sxm")
    # My_SXM.get_data_test(filename)
    # print(My_SXM.get_informations("Test.sxm"))
    # plt.imshow(My_SXM.get_data("Test.sxm"))
    # plt.show()

    evaluateLog()
