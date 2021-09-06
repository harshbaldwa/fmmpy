import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import pickle

t_design = {}

url = "http://neilsloane.com/sphdesigns/dim3/"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

harsh = soup.find_all("a")[::4]
for a in harsh[:6]:
    result = re.search('des.3.(.*).*.txt', a["href"])
    data = result.group(1).split(".")
    new_url = url + a["href"]
    new_url = new_url.replace(" ", "")
    new_page = requests.get(new_url)
    txt = new_page.text.split("\n")[:-1]
    array = np.array(txt, dtype=np.float32)
    data = np.array(data, dtype=np.int32)

    if data[0] in t_design:
        if t_design[data[0]]["order"] < data[1]:
            t_design[data[0]] = {"array": array, "order": data[1]}
        else:
            continue
    else:
        t_design[data[0]] = {"array": array, "order": data[1]}

with open("../fmm/t_design.pickle", "wb") as f:
    pickle.dump(t_design, f, protocol=pickle.HIGHEST_PROTOCOL)
