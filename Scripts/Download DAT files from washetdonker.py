import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs

# ---- CONFIGURATION ---- #

# List of locations for which you want to download data
locations = [
    "Lauwersoog-haven", "Lauwersoog", "Hornhuizen"
]

# Years and months for which data should be downloaded
years = ["2023", "2024"]
months = ["01", "02", "12"]

# Base URL for the website where the data is located
base_url = 'https://www.washetdonker.nl/data/'

# Directory to save the downloaded files (adjust as needed)
download_root = "C:/Users/Farhan/Desktop/SQM Data/"

# ---- FUNCTIONS ---- #

def find_directories(url):
    """
    Find and return all directories on the given website.
    
    Args:
        url (str): The website URL to search for directories.
        
    Returns:
        set: A set of directory names found on the webpage.
    """
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        directories = set()
        
        # Extract directory names from the links (?dir= pattern)
        for link in links:
            href = link['href']
            parsed_url = urlparse(href)
            query_params = parse_qs(parsed_url.query)
            if 'dir' in query_params:
                directory_name = query_params['dir'][0]
                directories.add(directory_name)
                
        return directories
    else:
        print(f"Failed to retrieve webpage: {url}. Status code: {response.status_code}")
        return None


def download_dat_files(url, download_dir):
    """
    Downloads all .dat files from the given URL into the specified directory.
    
    Args:
        url (str): The URL to download data from.
        download_dir (str): Directory where downloaded files will be saved.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            if href and href.endswith('.dat'):
                dat_url = urljoin(url, href)
                dat_filename = os.path.join(download_dir, os.path.basename(dat_url))

                # Check if file already exists
                if not os.path.exists(dat_filename):
                    dat_response = requests.get(dat_url)
                    if dat_response.status_code == 200:
                        with open(dat_filename, "wb") as dat_file:
                            dat_file.write(dat_response.content)
                        print(f"Downloaded: {dat_filename}")
                    else:
                        print(f"Failed to download: {dat_url}")
                else:
                    print(f"File already exists, skipping: {dat_filename}")
    else:
        print(f"Failed to fetch page: {url}. Status code: {response.status_code}")


def main():
    """
    Main function to download .dat files from the website for specified locations and time periods.
    """
    print("Starting download process...")
    
    for location in locations:
        for year in years:
            for month in months:
                # Construct the URL for the location, year, and month
                target_url = f"{base_url}?dir={location}/{year}/{month}"
                
                # Specify the directory where the files will be saved
                download_dir = os.path.join(download_root, location)
                
                # Download .dat files from the constructed URL
                print(f"\nProcessing: {location} - {year}/{month}")
                download_dat_files(target_url, download_dir)
    
    print("\nDownload process completed.")


# ---- EXECUTION ---- #

if __name__ == "__main__":
    main()

