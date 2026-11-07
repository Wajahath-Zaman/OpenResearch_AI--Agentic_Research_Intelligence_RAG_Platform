import streamlit as st

from frontend.services.api_client import APIError


def render_paper_comparison(api):

    st.header("📊 Research Paper Comparison")

    st.write(
        "Select research papers and compare their objectives, "
        "methodologies, datasets, architectures and results."
    )

    # Fetch available documents
    try:
        with st.spinner("Loading research documents..."):
            documents = api.list_documents(limit=500)

        documents = documents or []

    except APIError as error:
        st.error(str(error))
        return

    # Only allow processed documents
    processed_documents = [
        doc for doc in documents
        if doc.get("status") == "processed"
    ]

    if len(processed_documents) < 2:
        st.warning(
            "You need at least two processed research papers "
            "to perform a comparison."
        )
        return

    # Map document IDs to titles
    document_map = {
        doc["id"]: doc.get("title", "Untitled Research Paper")
        for doc in processed_documents
    }

    st.subheader("Select Research Papers")

    selected_ids = st.multiselect(
        "Choose two or more papers",
        options=list(document_map.keys()),
        format_func=lambda doc_id: document_map[doc_id]
    )

    query = st.text_area(
        "Comparison instructions",
        value=(
            "Compare the research objectives, methodologies, "
            "datasets, model architectures, performance "
            "and limitations of these research papers."
        ),
        height=120
    )

    if st.button(
        "🔍 Compare Papers",
        use_container_width=True
    ):

        st.session_state.pop("comparison_result", None)

        if len(selected_ids) < 2:
            st.warning("Please select at least two papers.")
            return

        if not query.strip():
            st.warning("Please enter comparison instructions.")
            return

        with st.spinner(
            "Analysing research papers and generating comparison..."
        ):

            try:
                result = api.compare_papers(
                    query=query.strip(),
                    document_ids=selected_ids
                )

                if not isinstance(result, dict):
                    raise APIError(
                        "Unexpected response from comparison service."
                    )

                if not result.get("comparison"):
                    raise APIError(
                        "The comparison service returned no comparison."
                    )

                st.session_state["comparison_result"] = result

            except APIError as error:
                st.error(str(error))
                return

    # Display comparison
    result = st.session_state.get("comparison_result")

    if not result:
        return

    st.divider()

    st.subheader("Comparison Results")

    st.markdown(result["comparison"])

    # Individual paper analyses
    analyses = result.get("comparison_results", [])

    if analyses:

        with st.expander("View Individual Paper Analyses"):

            for paper in analyses:

                st.markdown(
                    f"### Paper {paper.get('paper_number', 'N/A')}"
                )

                st.write(
                    f"**Document:** "
                    f"{document_map.get(paper.get('document_id'), 'Unknown')}"
                )

                st.markdown(
                    paper.get("analysis", "Analysis unavailable.")
                )

                citations = paper.get("citations", [])

                if citations:
                    st.write("**Sources:**")

                    for citation in citations:
                        st.write(citation)
