import ast
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_summary(url):
    """
    Scrapes the main content of a Wikipedia page and creates a 5-6 sentence summary.
    
    Args:
        url (str): URL of the Wikipedia page.
    
    Returns:
        str: A concise summary of the page.
    """
    # Request the page content
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main content (first few paragraphs)
    paragraphs = soup.select("div.mw-parser-output > p")
    if not paragraphs:
        return "No content found on the Wikipedia page."

    # Collect text from the first few paragraphs
    content = []
    for para in paragraphs:
        text = para.get_text(separator=" ", strip=True)  # Ensure spaces between inline elements
        if text:  # Avoid empty paragraphs
            content.append(text)
        if len(" ".join(content).split()) > 250:  # Stop when approximately 5-6 sentences are gathered
            break

    # Combine paragraphs and clean up extra whitespace
    summary = " ".join(content).strip()
    return summary


if __name__ == '__main__':
    wiki_urls = []

    with open('raw_wiki_entities_1733427824.841352.txt', 'r') as file:
        for i, line in enumerate(file):
            try:
                # Parse the line as a Python dictionary using ast.literal_eval
                record = ast.literal_eval(line.strip())

                # Ensure the record contains the required keys
                if 'wd_id' in record and 'wikipediaURL' in record:
                    wiki_urls.append({'wd_id': record['wd_id'], 'url': record['wikipediaURL']})
            except (ValueError, SyntaxError) as e:  # Handle malformed lines
                continue

    with open('wiki_summaries.csv', 'w') as file:
        for url in wiki_urls:
            summary = scrape_wikipedia_summary(url['url'])
            print("Wikipedia Summary:")
            print(summary)
            file.write(f"{url['wd_id']},{url['url']},'{summary}'\n")
