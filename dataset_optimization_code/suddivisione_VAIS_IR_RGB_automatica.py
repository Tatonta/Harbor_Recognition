import os
import re
import shutil


files = []
folders = []
FILE_PATH = 'F:\\Università\\secondo anno\\primo semestre\\Caponnetto\\PMC 2021 Xibilia Caponetto\\VAIS_ex'
FILE_PATH_IR = 'F:\\Università\\secondo anno\\primo semestre\\Caponnetto\\PMC 2021 Xibilia Caponetto\\VAIS_IR'
FILE_PATH_RGB = 'F:\\Università\\secondo anno\\primo semestre\\Caponnetto\\PMC 2021 Xibilia Caponetto\\VAIS_RGB'
# os.mkdir(FILE_PATH_IR), os.mkdir(FILE_PATH_RGB)
file_IR = {}
file_RGB = {}
def listDir(dir):
    for currentPath, dirNames, files in os.walk(dir):
        a = os.path.relpath(currentPath, start=FILE_PATH)
        if a == ".":
            continue
        else:
            for file in files:
                if re.match(".*[\d_\-]{10,}-eo.*", file):
                    if a not in file_RGB:
                        file_RGB[a] = []
                    file_RGB[a].append(file)
                else:
                    if a not in file_IR:
                        file_IR[a] = []
                    file_IR[a].append(file)


def createFiles(dir, folders):
    for folder, files in folders.items():
        os.mkdir(os.path.join(dir, folder))
        for file in files:
            shutil.copy(os.path.join(FILE_PATH, folder+"\\"+file), os.path.join(dir,folder+"\\"+file))


listDir(FILE_PATH)
print(file_IR)
print(file_RGB)
createFiles(FILE_PATH_RGB, file_RGB)
createFiles(FILE_PATH_IR, file_IR)
