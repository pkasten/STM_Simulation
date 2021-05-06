from pySPM.SXM import SXM
import matplotlib.pyplot as plt
import numpy as np
import struct, os, SXM_info

class My_SXM():

    @staticmethod
    def write_header(filename):
        with open(filename, "w") as file:
            settings = SXM_info.get_header_info()
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