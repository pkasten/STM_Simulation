import configparser as cp

conf = cp.ConfigParser()

conf['DEFAULT'] = {'lending_period' : 0, 'max_value': 0}
#conf['Fred'] = {'max_value': 200}
#conf = {'lendig_period': 30}

with open('toolhire.ini', 'w') as toolhire:
    conf.write(toolhire)

#conf.read('toolhire.ini')
#print(conf['DEFAULT']['max_value'])

#print("Done");