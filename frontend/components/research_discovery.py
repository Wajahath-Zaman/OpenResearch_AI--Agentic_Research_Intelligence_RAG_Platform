import streamlit as st

from frontend.services.api_client import APIError


def render_research_discovery(api):

    st.header("📚 Research Discovery")

    st.write(
        "Discover scientific publications using the OpenAlex research database."
    )

    with st.form("research_discovery_form"):

        query = st.text_input(
            "Research topic",
            placeholder="e.g. Large Language Models in Healthcare"
        )

        col1, col2 = st.columns(2)

        with col1:
            per_page = st.number_input(
                "Number of papers",
                min_value=1,
                max_value=100,
                value=10
            )

        with col2:
            save_to_database = st.checkbox(
                "Save discovered papers to database",
                value=True
            )

        submitted = st.form_submit_button(
            "🔍 Discover Research",
            use_container_width=True
        )

    if submitted:

        st.session_state.pop("research_result", None)

        if not query.strip():
            st.warning("Please enter a research topic.")

        else:
            with st.spinner("Searching OpenAlex..."):

                try:
                    result = api.discover_research(
                        query=query.strip(),
                        per_page=int(per_page),
                        save_to_database=save_to_database
                    )

                    if not isinstance(result, dict):
                        raise APIError(
                            "The backend returned an unexpected response."
                        )

                    st.session_state["research_result"] = result

                except APIError as error:
                    st.error(str(error))

                except Exception as error:
                    st.error(
                        f"Unexpected error during research discovery: {error}"
                    )

    result = st.session_state.get("research_result")

    if not result:
        return

    st.divider()
    st.subheader("Search Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Results", result.get("total_count", 0))
    col2.metric("Returned", result.get("returned_count", 0))
    col3.metric("Saved", result.get("saved_count", 0))
    col4.metric("Duplicates", result.get("duplicate_count", 0))

    papers = result.get("results") or []

    if not papers:
        papers = result.get("saved_documents") or []

    if not papers:
        st.info(
            "No research papers were found for this search. "
            "Try a different research topic."
        )
        return

    st.caption(f"Displaying {len(papers)} research papers.")

    for index, paper in enumerate(papers, start=1):

        title = paper.get("title") or "Untitled Research Paper"

        with st.expander(f"{index}. {title}"):

            authors = paper.get("authors") or []

            if authors:
                st.write(
                    "**Authors:** " + ", ".join(
                        str(author) for author in authors
                    )
                )

            st.write(
                f"**Publication Year:** "
                f"{paper.get('publication_year') or 'Not available'}"
            )

            if paper.get("doi"):
                st.write(f"**DOI:** {paper['doi']}")

            st.write("**Abstract:**")

            abstract = paper.get("abstract")

            if abstract:
                st.write(abstract)
            else:
                st.caption("Abstract unavailable.")

            if paper.get("source_url"):
                st.link_button(
                    "View Research Paper",
                    paper["source_url"]
                )
