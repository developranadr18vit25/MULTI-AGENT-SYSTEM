from agents import build_reader_agent , build_search_agent , writer_chain , critic_chain

def run_research_pipeline(topic:str)->dict:

    state={}

    print("Search agent working")

    search_agent=build_search_agent()

    search_result=search_agent.invoke({
        "messages":[{"role":"user" , "content":topic}]
    })

    state['search_results']=search_result["messages"][-1].content

    print("\n Search Result : ", state["search_results"])

    print("\n Now Reader Agent working")

    reader_agent=build_reader_agent()

    reader_result=reader_agent.invoke({
        "messages":[{"role":"user" , "content": f"Based on the search results about the topic {topic } , pick the most relevant and scrape it for deeper Content . Search results : {state['search_results'][:800]}"}]
    })

    state['scraped_content']=reader_result["messages"][-1].content

    print("Scraped Content : \n " , state['scraped_content'])

    research_combined=(
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state['report']=writer_chain.invoke({
        "topic":topic,
        "research":research_combined
    })

    print("Final report : \n " , state['report'])
    print("Critic chain reviews the report")

    state['feedback']=critic_chain.invoke({
        "report":state['report']
    })

    print("Critic report is \n : ", state['feedback'])


user_input=input("Enter a research topic : ")
run_research_pipeline(user_input)
