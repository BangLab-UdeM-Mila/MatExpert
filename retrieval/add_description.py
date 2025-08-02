from utils import *
import os

if __name__ == '__main__':
    set_list = ['val', 'test']

    for set_name in set_list:
        file_name = f'../data/basic/{set_name}.csv'
        # add_description(file_name)
        add_description_from_id(file_name)