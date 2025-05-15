import matplotlib.pyplot as plt
from matplotlib import font_manager

fontPath = "/usr/share/fonts/MyFonts/simhei.ttf"

font = font_manager.fontManager.addfont(path=fontPath)

for fn in font_manager.fontManager.ttflist:
    print(fn.name)