from pySPM.SXM import SXM
import matplotlib.pyplot as plt
import numpy as np
import struct, os, SXM_info
import Configuration as cfg

class My_SXM():

    @staticmethod
    def write_header(filename):
        with open(filename, "w") as file:
            settings = SXM_info.get_header_arr()
            for elem in settings:
                arg = elem[1]
                string = ""
                try:
                    arg = arg[0]
                except IndexError:
                    file.write(":{}:\n\n".format(elem[0]))
                    continue
                if len(elem[1]) == 1:
                    string = "\t".join(arg)
                    file.write(":{}:\n{}\n".format(elem[0], string))
                    continue
                else:
                    file.write(":{}:\n".format(elem[0]))
                    for arg in elem[1]:
                        file.write("{}\n".format("\t".join(arg)))

    @staticmethod
    def write_sxm(filename, data):

        #print(SXM_info.get_time())
        SXM_info.adjust_to_image(data, filename)
        #print(SXM_info.get_time())
        #print(SXM_info.get_header_dict()["REC_TIME"])
        try:
            with open(filename, "w") as file:
                file.write("")
        except FileNotFoundError:
            os.mkdir(cfg.get_sxm_folder())

        My_SXM.write_header(filename)
        My_SXM.write_image(filename, data)


    @staticmethod
    def write_image(filename, image):

        ang_per_bright = cfg.get_max_height().ang * 1e-10 / 255
        newmat = np.zeros(np.shape(image))

        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                newmat[i, j] = ang_per_bright * image[i, j]

        with open(filename, "ab") as file:
            file.write(b'\n')
            file.write(b'\x1a')
            file.write(b'\x04')

            header = SXM_info.get_header_dict()

            size = dict(pixels={
                'x': int(header['SCAN_PIXELS'][0][0]),
                'y': int(header['SCAN_PIXELS'][0][1])
            }, real={
                'x': float(header['SCAN_RANGE'][0][0]),
                'y': float(header['SCAN_RANGE'][0][1]),
                'unit': 'm'
            })
            im_size = size['pixels']['x'] * size['pixels']['y']
            #print(im_size)

            if header['SCANIT_TYPE'][0][1] == 'MSBFIRST':
                bitorder = '>'
            else:
                bitorder = '<'
            #print(bitorder)

            #length = str(im_size)
            #print(length)
            length = '1'

            if header['SCANIT_TYPE'][0][0] == 'FLOAT':
                d_type = 'f'
            elif header['SCANIT_TYPE'][0][0] == 'INT':
                d_type = 'i'
            elif header['SCANIT_TYPE'][0][0] == 'UINT':
                d_type = 'I'
            elif header['SCANIT_TYPE'][0][0] == 'DOUBLE':
                d_type = 'd'
            else:
                print("Error reading SCANIT_TYPE. Unexpected: {}".format(header['SCANIT_TYPE'][0][0]))
                d_type = 'f'
            #print(d_type)

            #print("Shape than: {}".format(np.shape(image)))
            data = newmat.reshape(im_size,)
            #print("Shape now: {}".format(np.shape(data)))

            format = bitorder + length + d_type
            #print(format)
            calced_size = struct.calcsize(format)
            #print(calced_size)


            for elem in data:
                file.write(struct.pack(format, elem))









    @staticmethod
    def get_data_test(filename):
        assert os.path.exists(filename)
        f = open(filename, 'rb')
        l = ''
        key = ''
        header = {}
        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    header[key].append(l.decode('ascii').split())

        while f.read(1) != b'\x1a':
            pass
        assert f.read(1) == b'\x04'
        assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
        data_offset = f.tell()
        size = dict(pixels={
            'x': int(header['SCAN_PIXELS'][0][0]),
            'y': int(header['SCAN_PIXELS'][0][1])
        }, real={
            'x': float(header['SCAN_RANGE'][0][0]),
            'y': float(header['SCAN_RANGE'][0][1]),
            'unit': 'm'
        })

        im_size = size['pixels']['x'] * size['pixels']['y']

        data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[
                                          header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))
        print(struct.unpack('>' + str(im_size) + 'f',
                                      f.read(4 * im_size)))

        data = np.flipud(data)
        return data


    @staticmethod
    def get_informations(filename):
        assert os.path.exists(filename)
        f = open(filename, 'rb')
        l = ''
        key = ''
        header = {}
        ret_str = []
        ret_str.append("Header_Information for File {}".format(filename))
        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    header[key].append(l.decode('ascii').split())

        ret_str.append("Key: {}".format(key))
        ret_str.append("header[key]: {}".format(header[key]))
        ret_str.append("header:")
        for x in header.keys():
            ret_str.append("{}: {}".format(x, header[x]))

        while f.read(1) != b'\x1a':
            pass
        assert f.read(1) == b'\x04'
        assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
        data_offset = f.tell()
        size = dict(pixels={
            'x': int(header['SCAN_PIXELS'][0][0]),
            'y': int(header['SCAN_PIXELS'][0][1])
        }, real={
            'x': float(header['SCAN_RANGE'][0][0]),
            'y': float(header['SCAN_RANGE'][0][1]),
            'unit': 'm'
        })

        im_size = size['pixels']['x'] * size['pixels']['y']

        data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))



        return "\n".join(ret_str)

    @staticmethod
    def get_data(filename, dontflip=False):
        assert os.path.exists(filename)
        f = open(filename, 'rb')
        l = ''
        key = ''
        header = {}
        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    header[key].append(l.decode('ascii').split())
        while f.read(1) != b'\x1a':
            pass
        assert f.read(1) == b'\x04'
        assert header['SCANIT_TYPE'][0][0] in ['FLOAT', 'INT', 'UINT', 'DOUBLE']
        data_offset = f.tell()
        size = dict(pixels={
            'x': int(header['SCAN_PIXELS'][0][0]),
            'y': int(header['SCAN_PIXELS'][0][1])
        }, real={
            'x': float(header['SCAN_RANGE'][0][0]),
            'y': float(header['SCAN_RANGE'][0][1]),
            'unit': 'm'
        })

        im_size = size['pixels']['x'] * size['pixels']['y']

        data = np.array(struct.unpack('<>'['MSBFIRST' == header['SCANIT_TYPE'][0][1]] + str(im_size) +
                                      {'FLOAT': 'f', 'INT': 'i', 'UINT': 'I', 'DOUBLE': 'd'}[header['SCANIT_TYPE'][0][0]],
                                      f.read(4 * im_size))).reshape((size['pixels']['y'], size['pixels']['x']))

        if not dontflip:
            data = np.flipud(data)


        return data


    @staticmethod
    def show_data(filename):
        #print(My_SXM.get_informations(filename))
        plt.imshow(My_SXM.get_data(filename))
        plt.show()

    #@staticmethod
    #def test():
