import csv
from dotenv import load_dotenv

load_dotenv()

from crewai_tools import FirecrawlScrapeWebsiteTool, MDXSearchTool, ScrapeWebsiteTool
from crew import PublicRiskAnalystCrew

urls = []

with open("domain_frequency.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        urls.append(row[0])


if __name__ == "__main__":
    web_url = f"https://{urls[0].replace('https://', '').replace('http://', '')}"

    result = PublicRiskAnalystCrew().run(
        inputs=dict(
            url=web_url,
        )
    )

    print(result.json_dict)

    # with open("report.json", "w") as f:
    #     f.write(result.raw)
