import streamlit as st

from frontend.services.api_client import APIError


def render_chat_interface(api):

    st.header("🤖 Research Assistant")

    st.caption(
        "Ask questions about your research papers or discover new research."
    )

    # Initialise conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    # Display previous messages
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("route"):
                st.caption(f"Agent route: {message['route'].upper()}")

            if message.get("citations"):
                with st.expander("View Sources"):
                    for citation in message["citations"]:
                        st.write(citation)

    # Chat input
    query = st.chat_input(
        "Ask something about your research..."
    )

    if query:

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Research assistant is thinking..."):

                try:
                    result = api.chat(query=query)

                    answer = result["answer"]
                    route = result["route"]
                    citations = result.get("citations", [])

                    st.markdown(answer)

                    st.caption(f"Agent route: {route.upper()}")

                    if citations:
                        with st.expander("View Sources"):
                            for citation in citations:
                                st.write(citation)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route,
                        "citations": citations
                    })

                except APIError as error:

                    st.error(str(error))

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Request failed: {error}"
                    })