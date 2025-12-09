import pandas as pd
import os
import sys

def add_files_in_folder(writer, folder):
    for element in os.listdir(folder):
        path = f"{folder}/{element}"
        if os.path.isdir(path):
            add_files_in_folder(writer, path)
        else:
            file_name_arr = element.split('.') 
            if file_name_arr [-1] == 'csv':
                name = file_name_arr[0]
                if name == "metrics":
                    name = name +'_'.join(path.split('/')[-2].split("_")[0:2]) + "".join([string[0] for string in path.split('/')[-2].split("_")[3:]])
                print("writing " + name)
                df = pd.read_csv(path)
                df.to_excel(writer,sheet_name=name)

folder = "data/experiments/" + sys.argv[1]

with pd.ExcelWriter(f"{folder}/{sys.argv[1].split("/")[-1]}.xlsx", engine="xlsxwriter", mode="w") as writer:
    add_files_in_folder(writer,folder)












