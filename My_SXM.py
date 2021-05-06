from pySPM.SXM import SXM
import matplotlib.pyplot as plt
import numpy as np
import struct, os

class My_SXM():

    @staticmethod
    def get_informations(filename):
        assert os.path.exists(filename)
        f = open(filename, 'rb')
        l = ''
        key = ''
        header = {}
        ret_str = "Header_Information for File {}\n".format(filename)

        while l != b':SCANIT_END:':
            l = f.readline().rstrip()
            if l[:1] == b':':
                key = l.split(b':')[1].decode('ascii')
                header[key] = []
            else:
                if l:  # remove empty lines
                    header[key].append(l.decode('ascii').split())

        ret_str.append("Key: {}".format(key))
        print("header[key]: {}".format(header[key]))
        print("header:")
        for x in header.keys():
            print("{}: {}".format(x, header[x]))

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



        return data

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