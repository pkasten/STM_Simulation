import os

mv_from = ""
mv_to = ""
namescheme = ""


# returns path to maximum Indexed file. Namescheme Index##### where # means a number
def maxIndexFile(folder, namescheme):
    presuf = str(namescheme).split('.', maxsplit=1)
    prefix = presuf[0]
    suffix = presuf[1]
    placeholder = (str(namescheme).find('#'), str(namescheme).rfind('#'))
    prefix_wo_place = presuf[0:placeholder[0]]
    if placeholder[0] == -1 or placeholder[1] == -1: raise ValueError

    if folder == None:
        folder = os.getcwd()
    files = os.listdir(folder)
    indexes = []
    # besser andere funktion, immer 1.?
    for file in files:
        index = int(file[placeholder[0]: placeholder[1]])
        indexes.append(index)

    return prefix_wo_place + str(max(indexes)) + "." + suffix
