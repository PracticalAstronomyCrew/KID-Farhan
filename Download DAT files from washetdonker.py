#!/usr/bin/env python
# coding: utf-8

# In[1]:


locations = [
    "tZandt",
    "ranking",
    "pdf",
    "mobiel",
    "cosine-Warmond",
    "ZwarteHaan",
    "Westpoort",
    "Westhoek",
    "Weerribben",
    "Vlieland-Oost",
    "Tolbert",
    "Texel",
    "Termunten",
    "Stadskanaal",
    "Sneek",
    "Sellingen",
    "Schiermonnikoog-dorp",
    "Schiermonnikoog",
    "Sappemeer",
    "SCENE-SallochyWood",
    "Roodeschool",
    "Rijswijk",
    "Oostkapelle",
    "Oldenburg",
    "ObsonWheels",
    "Norddeich-zwei",
    "Norddeich-eins",
    "Noordpolderzijl",
    "Natuurschuur-Terschelling",
    "Moddergat",
    "Lochem",
    "Lemmer",
    "Leiden-Sterrewacht",
    "Lauwersoog-haven",
    "Lauwersoog",
    "Katlijk",
    "Hulshorst",
    "Hornhuizen",
    "Hippolytushoef",
    "Heerenveen01",
    "Heerenveen-Station",
    "Harlingen01",
    "Haaksbergen",
    "Groningen-ZernikeCampus",
    "Groningen-DeHeld",
    "Griend",
    "Gorredijk",
    "Farmsum",
    "Erica",
    "Emden",
    "DenHelder",
    "Delft",
    "DeZilk",
    "Boschplaat",
    "Borkum-Ostland",
    "Borkum",
    "Boerakker",
    "Assen",
    "Amrum",
    "Ameland-Natuurcentrum-Nes",
    "Aldeboarn",
    "Akkrum"
]
locations = ["Lauwersoog-haven","Lauwersoog","Hornhuizen"]

years = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


# In[2]:


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs

# THIS CODE IS TO FIND THE DIRECTORIES IN THE WEBSITE


# Replace this URL with the website you want to search for directories
website_url = 'https://www.washetdonker.nl/data/'

# Send an HTTP GET request to the website
response = requests.get(website_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the links on the webpage
    links = soup.find_all('a', href=True)
    
    # Extract directory names from the links
    directories = set()
    for link in links:
        href = link['href']
        # Check if the link points to a directory (using the ?dir= pattern)
        parsed_url = urlparse(href)
        query_params = parse_qs(parsed_url.query)
        if 'dir' in query_params:
            directory_name = query_params['dir'][0]
            directories.add(directory_name)
    
    # Print the directory names found
    if directories:
        print("Directories found on the website:")
        for directory in directories:
            print(directory)
    else:
        print("No directories found on the website.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[2]:


import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs

def download_dat_files(url, download_dir):
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Send an HTTP GET request to fetch the HTML content of the page
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links on the page
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')
            
            # Check if the link ends with .dat (assuming all DAT files have this extension)
            if href and href.endswith('.dat'):
                dat_url = urljoin(url, href)
                dat_filename = os.path.join(download_dir, os.path.basename(dat_url))

                # Check if the file already exists in the download directory
                if not os.path.exists(dat_filename):
                    # Send an HTTP GET request to download the file
                    dat_response = requests.get(dat_url)

                    # Check if the request was successful (status code 200)
                    if dat_response.status_code == 200:
                        # Save the file in the download directory
                        with open(dat_filename, "wb") as dat_file:
                            dat_file.write(dat_response.content)
                        print(f"Downloaded: {dat_filename}")
                    else:
                        print(f"Failed to download: {dat_url}")
                else:
                    print(f"Skipping existing file: {dat_filename}")
    else:
        print(f"Failed to fetch page: {url}")

# Example usage:
years = ["2023", "2024"]
months = ["01", "02", "12"]

for i in locations:
    for j in years:
        for k in months:
            download_dat_files("https://www.washetdonker.nl/data/?dir="+i+"/"+j+"/"+k,
                               "C:/Users/Farhan/Desktop/SQM Data/"+i+"/")

