from datetime import datetime
import logging
import json
import ipdb
from collections import OrderedDict

def get_current_date_time(just_get_day=False):
    # datetime object containing current date and time
    now = datetime.now()
    if just_get_day:
        dt_string = now.strftime("%d_%m_%Y")
    else:
        dt_string = now.strftime("%d_%m_%Y_%H:%M")
    return dt_string

class json_logger:

    def __init__(self,dirname = 'OutFiles', filename='json_logger.json'):
        self.fname =   dirname + '/' + get_current_date_time() + '_' + filename
        json_object = OrderedDict()
        self.add(json_object)


    def add(self, values):
        json_object = values
        with open(self.fname, "w") as outfile:
            json.dump(json_object, outfile, indent = 4, ensure_ascii = False)

    def update(self, values):
        with open(self.fname, 'r') as openfile:
            json_object = json.load(openfile)

        json_object.update(values)
        with open(self.fname, 'w') as outfile:
            json.dump(json_object, outfile, indent = 4, ensure_ascii = False)


def logger_fname(dirname = 'OutFiles', filename='logger.txt', level=logging.DEBUG):
    fname =  dirname + '/' + get_current_date_time() + '_' + filename
    return fname



def write_config_files_json(variables, var_list, dirname = 'OutFiles', filename='json_config.txt'):
    jsonData = OrderedDict()
    for var in var_list:
        if var in variables:
            jsonData[var] = variables[var]

    filename =  dirname + '/' + get_current_date_time() + '_' + filename
    with open(filename, 'w') as outfile:
        json.dump(jsonData, outfile, sort_keys = True, indent = 4,
                  ensure_ascii = False)

    return jsonData

def write_config_files(variables, var_list, dirname = 'OutFiles', filename='config.txt'):
    variables = globals()
    filename =  dirname + '/' + get_current_date_time() + '_' + filename
    outfile = open(filename, 'w')
    outfile.write("********** START CONFIG VALUES***********")
    for var in var_list:
        if var in variables:
            outfile.write('{}={}\n'.format(var, str(variables[var])))

    outfile.write("**********END CONFIG VALUES***********")
    outfile.close()
    return
