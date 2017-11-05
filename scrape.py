import requests
from bs4 import BeautifulSoup
import json
import time

def get_page(url):
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
    return r.text

def get_jobs(url):
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")

    jobs = []

    for job in soup.find_all('div', class_='JobSearchCard-item'):
        obj = {}
        title = job.find('a', class_='JobSearchCard-primary-heading-link')
        obj['title'] = title.text.strip()

        bidbut = job.find('a', class_='JobSearchCard-ctas-btn')
        details_url = bidbut['href'].strip()
        if len(details_url) == 0 or details_url.endswith(".html"):
            continue
        obj['url'] = bidbut['href']

        obj['desc'] = get_job_details(base + obj['url'])

        jobs.append(obj)

    return jobs

def get_job_details(url):
    print("         doing url: " + url)
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")

    desc = ""
    details = soup.find('div', class_='PageProjectViewLogout-detail')
    if details is not None:
        for p in details.find_all('p', attrs={'class': None}):
            desc += p.text.strip()

    return desc

base = "https://www.freelancer.com"
#urls = [base + "/jobs/regions/?keyword=mobile&results=100"]
urls = []
for i in range(52, 79):
    urls.append(base + "/jobs/regions/" + str(i) + "/?keyword=mobile&results=100")

for idx, url in enumerate(urls):
    print("DOING MAIN URL:" + url)
    jobs = get_jobs(url)
    with open("data" + str(idx+1) + ".json", "w") as jsonfile:
        json.dump(jobs, jsonfile)
    time.sleep(30)
