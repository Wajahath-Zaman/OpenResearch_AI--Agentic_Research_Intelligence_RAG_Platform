import streamlit as st

from frontend.services.api_client import APIError


def render_document_management(api):

    st.header("📄 Document Management")

    st.write(
        "Upload, process, and manage your research documents."
    )

    # PDF Upload
    st.subheader("Upload Research Paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum file size: 20 MB"
    )

    if uploaded_file:

        st.write(f"Selected file: **{uploaded_file.name}**")

        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("File exceeds the 20 MB size limit.")

        elif st.button(
            "Upload PDF",
            use_container_width=True
        ):

            try:
                with st.spinner("Uploading document..."):
                    result = api.upload_document(uploaded_file)

                st.success("Document uploaded successfully!")

                if result:
                    st.json(result)

                st.rerun()

            except APIError as error:
                st.error(f"Upload failed: {error}")

    st.divider()

    # Document Listing
    st.subheader("Your Research Documents")

    try:
        with st.spinner("Loading documents..."):
            documents = api.list_documents(limit=500)

        documents = documents or []

    except APIError as error:
        st.error(f"Unable to load documents: {error}")
        return

    if not documents:
        st.info(
            "No documents found. Upload a research paper to begin."
        )
        return

    st.write(f"**Total Documents:** {len(documents)}")

    # Document status summary
    processed_count = sum(
        doc["status"] == "processed" for doc in documents
    )

    pending_count = sum(
        doc["status"] in ["discovered", "downloaded"]
        for doc in documents
    )

    failed_count = sum(
        doc["status"] == "failed" for doc in documents
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Processed", processed_count)
    col2.metric("Pending", pending_count)
    col3.metric("Failed", failed_count)

    st.divider()

    for document in documents:

        document_id = document["id"]
        title = document.get("title") or "Untitled Research Paper"
        status = document["status"]

        with st.expander(f"📄 {title}"):

            st.write(f"**Document ID:** `{document_id}`")

            st.write(
                f"**Publication Year:** "
                f"{document.get('publication_year') or 'Unknown'}"
            )

            st.write(f"**Status:** `{status.upper()}`")

            if document.get("doi"):
                st.write(f"**DOI:** {document['doi']}")

            st.divider()

            col1, col2 = st.columns(2)


            # Document actions
            with col1:

                if status == "discovered":

                    if document.get("file_path"):
                        st.info("PDF already available locally.")

                    elif document.get("pdf_url"):

                        if st.button(
                            "⬇️ Download PDF",
                            key=f"download_{document_id}",
                            use_container_width=True
                        ):

                            try:
                                with st.spinner("Downloading PDF..."):
                                    result = api.download_document(
                                        document_id
                                    )

                                st.success("PDF downloaded successfully!")
                                st.rerun()

                            except APIError as error:
                                st.error(f"Download failed: {error}")

                    else:
                        st.warning("No open-access PDF available.")

                elif status == "downloaded":

                    if st.button(
                        "⚙️ Process PDF",
                        key=f"ingest_{document_id}",
                        use_container_width=True
                    ):

                        try:
                            with st.spinner("Processing document..."):
                                result = api.ingest_document(
                                    document_id
                                )

                            st.success(
                                f"Successfully processed "
                                f"{result['total_chunks']} chunks."
                            )

                            st.rerun()

                        except APIError as error:
                            st.error(
                                f"Processing failed: {error}"
                            )

                elif status == "processed":
                    st.success("Document processed")

                elif status == "failed":

                    if st.button(
                        "🔄 Retry Processing",
                        key=f"retry_{document_id}",
                        use_container_width=True
                    ):

                        try:
                            with st.spinner("Retrying processing..."):
                                result = api.ingest_document(
                                    document_id
                                )

                            st.success("Document processed successfully!")
                            st.rerun()

                        except APIError as error:
                            st.error(str(error))

            # Document deletion
            with col2:

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{document_id}",
                    use_container_width=True
                ):
                    st.session_state[
                        f"confirm_delete_{document_id}"
                    ] = True

                if st.session_state.get(
                    f"confirm_delete_{document_id}", False
                ):

                    st.warning(
                        "Are you sure you want to delete this document?"
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "Confirm",
                            key=f"confirm_{document_id}"
                        ):

                            try:
                                api.delete_document(document_id)

                                st.session_state.pop(
                                    f"confirm_delete_{document_id}",
                                    None
                                )

                                st.success("Document deleted.")
                                st.rerun()

                            except APIError as error:
                                st.error(
                                    f"Deletion failed: {error}"
                                )

                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"cancel_{document_id}"
                        ):
                            st.session_state.pop(
                                f"confirm_delete_{document_id}",
                                None
                            )
                            st.rerun()
