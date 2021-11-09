import re

import requests
import yaml
from bs4 import BeautifulSoup

t_design = {}

url = "http://neilsloane.com/sphdesigns/dim3/"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

harsh = soup.find_all("a")[::4]
for a in harsh:
    result = re.search('des.3.(.*).*.txt', a["href"])
    data = result.group(1).split(".")
    new_url = url + a["href"]
    new_url = new_url.replace(" ", "")
    new_page = requests.get(new_url)
    txt = new_page.text.split("\n")[:-1]
    array = list(map(float, txt))
    data = list(map(int, data))
    print(data[0], " - ", data[1])

    if data[0] in t_design:
        if t_design[data[0]]["order"] < data[1]:
            t_design[data[0]] = {"array": array, "order": data[1]}
        else:
            continue
    else:
        t_design[data[0]] = {"array": array, "order": data[1]}

with open('fmm/t_design.yaml', 'w') as outfile:
    yaml.dump(t_design, outfile)
