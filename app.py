import base64
import os

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()
FASTAPI_URL = os.getenv("FASTAPI_URL")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_document" not in st.session_state:
    st.session_state.research_document = None
if "active_agents" not in st.session_state:
    st.session_state.active_agents = {
        "snowflake_agent": False,
        "rag_agent": False,
        "web_search_agent": False,
    }

st.set_page_config(page_title="LangGraph NVIDIA Research Agent", layout="wide")

# Sidebar
with st.sidebar:
    st.title("LangGraph NVIDIA Research Agent")
    st.markdown(
        "This assistant uses multiple agents to generate research documents based on NVIDIA data:"
    )
    st.markdown(
        "- **Snowflake Agent**: Analyzes NVIDIA valuation measures, generates summary and visualization charts"
    )
    st.markdown(
        "- **RAG Agent**: Processes NVIDIA 10-K/10-Q reports (2022-2025) with hybrid search using Pinecone"
    )
    st.markdown("- **Web Search Agent**: Retrieves real-time data from web")

    st.subheader("Select Active Agents")

    st.session_state.active_agents["snowflake_agent"] = st.checkbox(
        "Snowflake Agent (Valuation Measures)",
        value=st.session_state.active_agents["snowflake_agent"],
        help="Analyzes NVIDIA valuation measures, generates summary and visualization charts using Snowflake database",
    )

    st.session_state.active_agents["rag_agent"] = st.checkbox(
        "RAG Agent (10-K/Q Reports)",
        value=st.session_state.active_agents["rag_agent"],
        help="Processes NVIDIA 10-K/Q reports from 2022-2025",
    )

    st.session_state.active_agents["web_search_agent"] = st.checkbox(
        "Web Search Agent (Tavily)",
        value=st.session_state.active_agents["web_search_agent"],
        help="Retrieves latest information from the web using Tavily",
    )

    # Add year/quarter filters
    st.subheader("Report Filters")
    if "year" not in st.session_state:
        st.session_state.year = ""
    if "quarter" not in st.session_state:
        st.session_state.quarter = ""

    st.session_state.year = st.selectbox(
        "Fiscal Year", options=["None", "2022", "2023", "2024", "2025"], index=1
    )
    if st.session_state.year == "None":
        st.session_state.quarter = st.selectbox(
            "Fiscal Quarter",
            options=["None", "1", "2", "3", "4"],
            index=0,
            disabled=True,
        )
    else:
        st.session_state.quarter = st.selectbox(
            "Fiscal Quarter",
            options=["None", "1", "2", "3", "4"],
            index=0,
        )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.research_document = None
        st.rerun()

# Main chat interface
st.title("LangGraph NVIDIA Research Assistant")


# Function to create a download link for the markdown file
def get_download_link(file_content, file_name):
    """
    Generates a link allowing the user to download a file from the app.
    :param file_content: The content of the file to download.
    :param file_name: The name of the file to download.
    :return: A link to download the file.
    """
    b64 = base64.b64encode(file_content.encode()).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href


# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if (
            message["role"] == "assistant"
            and "is_markdown" in message
            and message["is_markdown"]
        ):
            st.markdown(message["content"])
            if "visualization" in message:
                st.image(
                    base64.b64decode(message["visualization"]), use_column_width=True
                )
        else:
            st.write(message["content"])

# Show which agents are currently active
active_agent_names = [
    name for name, is_active in st.session_state.active_agents.items() if is_active
]
if active_agent_names:
    st.info(f"Active agents: {', '.join(active_agent_names)}")
else:
    st.warning("No agents selected. Please enable at least one agent in the sidebar.")


# Chat input - only enable if at least one agent is active
if len(active_agent_names) > 0:
    if prompt := st.chat_input("Ask a question about NVIDIA..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response with a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner(
                f"Generating research document using {', '.join(active_agent_names)} agents..."
            ):
                # Call the FastAPI endpoint to generate the research document
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/chat",
                        params={
                            "message": prompt,
                            "snowflake_agent": st.session_state.active_agents[
                                "snowflake_agent"
                            ],
                            "pinecone_agent": st.session_state.active_agents[
                                "rag_agent"
                            ],
                            "websearch_agent": st.session_state.active_agents[
                                "web_search_agent"
                            ],
                            "year": (
                                None
                                if st.session_state.year == "None"
                                else int(st.session_state.year)
                            ),
                            "quarter": (
                                None
                                if st.session_state.quarter == "None"
                                else int(st.session_state.quarter)
                            ),
                        },
                        timeout=120,  # Increased timeout for complex queries
                    )

                    if response.status_code == 200:
                        print("Response: ", response.json())
                        research_document = response.json().get("report", "")

                        # visualizations = response.json().get("visualizations", [])

                        # Store the research document in session state
                        st.session_state.research_document = research_document

                        # Display the markdown content
                        st.markdown(research_document)

                        # Display visualization if available
                        # st.title("Visualizations")
                        # for viz, viz_summary in visualizations:
                        #     try:
                        #         st.subheader(viz_summary)
                        #         exec(viz)
                        #     except Exception as e:
                        #         continue

                        # Add download button for the research document
                        st.markdown(
                            get_download_link(
                                research_document, "Research_Document.md"
                            ),
                            unsafe_allow_html=True,
                        )

                        # Add assistant response to chat history with visualization if available
                        message_data = {
                            "role": "assistant",
                            "content": research_document,
                            "is_markdown": True,
                        }

                        st.session_state.messages.append(message_data)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"Error: Unable to generate research document. Status code: {response.status_code}",
                                "is_markdown": False,
                            }
                        )
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Error: Unable to connect to the research API. {str(e)}",
                            "is_markdown": False,
                        }
                    )
else:
    st.chat_input("Please select at least one agent to continue...", disabled=True)

# Display download button for the latest research document if it exists
if st.session_state.research_document:
    st.sidebar.markdown("### Download Latest Research")
    st.sidebar.markdown(
        get_download_link(st.session_state.research_document, "Research_Document.md"),
        unsafe_allow_html=True,
    )


#     user_input = st.chat_input("Ask the agent a question...")

#     if user_input:
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
#         with st.spinner("Thinking..."):
#             response = requests.post(f"{FASTAPI_URL}/chat", json={
#                 "query": user_input,
#                 "chat_history": st.session_state.chat_history
#             })
#             data = response.json()
#             agent_reply = data.get("response", "No response from agent.")
#             st.session_state.chat_history.append({"role": "agent", "content": agent_reply})


#     st.subheader("📥 Download Latest Report")
#     if st.button("Download Markdown Report"):
#         download_response = requests.get(f"{FASTAPI_URL}/download-report")
#         if download_response.status_code == 200:
#             st.download_button(
#                 label="📄 Download Markdown File",
#                 data=download_response.content,
#                 file_name="nvidia_report.md",
#                 mime="text/markdown"
#             )
#         else:
#             st.error("Failed to download report.")

# Tabs for different tools
# tab1, tab2, tab3 = st.tabs(["🔍 Search Pinecone", "🌐 Web Search", "💬 Chat Agent"])

# # --- Search Pinecone Tool ---
# with tab1:
#     st.header("🔍 Pinecone Document Search")
#     query = st.text_input("Enter your query")
#     year = st.text_input("Year (optional)")
#     quarter = st.text_input("Quarter (optional)")
#     top_k = st.slider("Top K Results", 1, 10, 5)

#     if st.button("Search Pinecone"):
#         response = requests.post(f"{FASTAPI_URL}/run-tool", json={
#             "tool": "search_pinecone",
#             "query": query,
#             "year": year,
#             "quarter": quarter,
#             "top_k": top_k
#         })
#         result = response.json()
#         st.subheader("Results")
#         st.text(result.get("result", "No result returned"))

# # --- Web Search Tool ---
# with tab2:
#     st.header("🌐 Web Search Tool")
#     web_query = st.text_input("Enter your web search query")
#     if st.button("Search Web"):
#         response = requests.post(f"{FASTAPI_URL}/run-tool", json={
#             "tool": "web_search",
#             "query": web_query
#         })
#         result = response.json()
#         st.subheader("Web Results")
#         st.text(result.get("result", "No result returned"))

# # --- Chat Interface ---
# with tab3:
#     st.header("💬 Chat with the Research Agent")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     user_input = st.chat_input("Ask the agent a question...")

#     if user_input:
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
#         with st.spinner("Thinking..."):
#             response = requests.post(f"{FASTAPI_URL}/chat", json={
#                 "query": user_input,
#                 "chat_history": st.session_state.chat_history
#             })
#             data = response.json()
#             agent_reply = data.get("response", "No response from agent.")
#             st.session_state.chat_history.append({"role": "agent", "content": agent_reply})

#     for msg in st.session_state.chat_history:
#         if msg["role"] == "user":
#             st.chat_message("user").write(msg["content"])
#         else:
#             st.chat_message("assistant").write(msg["content"])

#     st.subheader("📥 Download Latest Report")
#     if st.button("Download Markdown Report"):
#         download_response = requests.get(f"{FASTAPI_URL}/download-report")
#         if download_response.status_code == 200:
#             st.download_button(
#                 label="📄 Download Markdown File",
#                 data=download_response.content,
#                 file_name="nvidia_report.md",
#                 mime="text/markdown"
#             )
#         else:
#             st.error("Failed to download report.")
