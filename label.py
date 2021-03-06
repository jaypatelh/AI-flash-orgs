import argparse
import os
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("dir", help="a directory containing dataX.json files")
parser.add_argument("start", help="file number to start with")
args = parser.parse_args()

reg = r'data(\d+).json'

labels = ["Lo-Fi UI Sketches", "UX Testing", "Interactive mockup", "Visual Design", "Stress Testing", "Schema Design", "Database Setup", "Core backend logic", "Provide public API", "Integrate external API calls", "Front-end Code", "Deploy", "Data privacy compliance", "Set up user authentication", "Speed optimization", "User testing"]

labels_print = '\n'.join([str(i) + ": " + s for i, s in enumerate(labels)])

output_file = "labels-output.tsv"

cnt = 0

for root, dirs, files in os.walk(args.dir):
    for f in files:
        f_no = int(re.match(reg, f).group(1))
        if f_no < int(args.start):
            continue
        path = root + "/" + f

        with open(path, "r") as datfile:
            data = json.load(datfile)
            for elem in data:
                print("=> LABELED SO FAR: " + str(cnt))
                print()
                print(elem['title'])
                print("-" * len(elem['title']))
                print(elem['desc'])
                print()
                print("skills:", elem['skills'])
                print("===================================")
                print(labels_print)
                print()
                text = input("=> Enter matching label numbers comma-separated (blank to pass): ")  # Python 3
                print("^ was " + path + ", " + elem['url'])
                if text != "":
                    cnt += 1
                text = "-" if text == "" else text
                with open(output_file, "a") as out:
                    out.write(elem['url'] + "\t" + text + "\t" + str(f_no) + "\n")
                _=os.system("clear")