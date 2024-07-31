from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_cohere import ChatCohere


from crewai_tools import (
    FirecrawlScrapeWebsiteTool,
    MDXSearchTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
    FileReadTool,
)

llm = ChatCohere()

# scrape_tool = FirecrawlScrapeWebsiteTool(
#     api_key="fc-59eb5af8a165486ba2413465e4d9aaf3",
# )

web_tool = WebsiteSearchTool(
    # config=dict(
    # llm=dict(provider="", config=dict(model="mixtral-8x7b-32768")),
    # embedder=dict(provider="google", config=dict(model="models/embedding-001")),
    # ),
)

file_tool = FileReadTool(
    file_path="public_risk.txt",
    # config=dict(
    #     llm=dict(provider="groq", config=dict(model="mixtral-8x7b-32768")),
    #     embedder=dict(provider="google", config=dict(model="models/embedding-001")),
    # ),
)


class PublicRiskAnalystCrew:

    def search_agent(self) -> Agent:
        return Agent(
            role="Website privacy policy and terms and conditions extractor",
            goal="Given a website URL, extract the privacy policy, terms and conditions, etc, pages from the website.",
            backstory="A highly skilled professional that can extract the privacy policy, terms and conditions, etc, pages from any website.",
            tools=[web_tool],
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )

    def pp_tc_search_task(self) -> Task:
        return Task(
            description="For the given {url} website, use a Search in a specific website tool to find and extract the privacy policy, terms and conditions, etc pages.",
            expected_output="All the content from the privacy policy, terms and conditions, etc, pages for the given website",
            agent=self.search_agent(),
        )

    def public_risk_analyst(self) -> Agent:
        return Agent(
            role="Public risk analyzer that can read through extracted websites data and determine the public risk of a company.",
            goal="""
            Read the extracted data then read the public_risk.txt policies using the file tool, then find all the questions in the file and answer those questions from 
    the scraped data to determine the public risk of the company.
            """,
            backstory="Compliance officer at a company that needs to determine the public risk of a company.",
            tools=[file_tool],
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )

    def public_risk_task(self) -> Task:
        return Task(
            description="This agent is responsible for analyzing the public risk from the already provided scraped data. Use a file reading tool and read the public_risk.txt file and then for all the questions in categories and sub-categories answer policy questions in yes/no from the scraped data. DO NOT LEAVE any questions. ANSWER ALL",
            expected_output="The agent should return array of JSON objects, where an object contains 'question' (i.e. the policy to be looked for), 'answer' (i.e. Yes or No), 'category' (i.e. Category of the question answered) keys with their respective values. ANSWER EVERY QUESTION. DO NOT LEAVE OR TRIM ANY OUTPUT",
            agent=self.public_risk_analyst(),
        )

    def run(self, inputs):
        crew = Crew(
            agents=[self.search_agent(), self.public_risk_analyst()],
            tasks=[self.pp_tc_search_task(), self.public_risk_task()],
            verbose=True,
            process=Process.sequential,
        )

        result = crew.kickoff(inputs=inputs)
        return result
