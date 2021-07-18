import ConfigParser
parser = ConfigParser.ConfigParser()

info = {}

parser.add_section('User-Info')
for key in info.keys():
    parser.set('User-Info', key, info[key])

with open('config.ini', 'w') as f:
    parser.write(f)