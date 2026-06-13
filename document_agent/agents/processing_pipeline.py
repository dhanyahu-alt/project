from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool

from ..util.settings import MODEL_FLASH

from ..tools.pdf_loader_tool import load_pdf, render_page_as_image
from ..tools.ocr_tool import extract_text_from_image_bytes

from ..tools.state_tool import update_processing_stage

from .classification_agent    import classification_agent
from .validation_agent        import validation_agent

from .loa_extraction_agent    import loa_extraction_agent
from .notice_extraction_agent import notice_extraction_agent
from .business_doc_agent      import business_doc_agent

loa_tool      = AgentTool(agent=loa_extraction_agent)
notice_tool   = AgentTool(agent=notice_extraction_agent)
business_tool = AgentTool(agent=business_doc_agent)


document_loader_agent = LlmAgent(
    name="document_loader_agent",
    model=MODEL_FLASH,
    tools=[
        load_pdf,
        render_page_as_image,
        extract_text_from_image_bytes,
        update_processing_stage,
    ],
    instruction="""You are the document loader stage of the processing pipeline.
    Your job is to load a PDF document and extract its full text content.

    The document file path is stored in session state under the key
    current_doc_path. Read this value at the start.

    LOADING STEPS:

    Step 1 -- Call update_processing_stage with LOADING.

    Step 2 -- Call load_pdf with the file path from current_doc_path.

    If load_pdf returns is_success as False:
        Write the error message to session state key pipeline_error.
        Write True to session state key pipeline_failed.
        Stop and report the error.

    If load_pdf returns is_success as True:
        Note the file_name, page_count, text, and is_scanned values.

    Step 3 -- If is_scanned is True:
        The PDF contains scanned images and has no embedded text.
        For each page (up to the page_count):
            Call render_page_as_image with the file path and page number.
            Call extract_text_from_image_bytes with the returned image bytes.
            Collect the extracted_text from each page result.
        Combine all page texts into one full document text.

    If is_scanned is False:
        Use the text field from the load_pdf result directly.

    Step 4 -- Write the following to session state:
        Write the full extracted text to session state key raw_text.
        Write the page_count value to session state key page_count.
        Write the is_scanned value to session state key is_scanned.
        Write the file_name value to session state key file_name.
        Write False to session state key pipeline_failed.

    Step 5 -- Call update_processing_stage with LOADED.

    IMPORTANT:
        Do not attempt classification or extraction -- that is handled
        by the next agents in the pipeline.
        Do not use curly brace template syntax in any response.
        Write all output to session state as described in Step 4.
    """,
)

extraction_router_agent = LlmAgent(
    name="extraction_router_agent",
    model=MODEL_FLASH,
    tools=[
        loa_tool,
        notice_tool,
        business_tool,
        update_processing_stage,
    ],
    instruction="""You are the extraction routing stage of the processing pipeline.
    Your job is to read the classification result from session state and
    call the correct extraction tool for the identified document type.

    ROUTING STEPS:

    Step 1 -- Read the doc_type value from session state key doc_type.
              Read the document text from session state key raw_text.

    If doc_type is not set or is empty:
        Write an error to session state key pipeline_error.
        Write True to session state key pipeline_failed.
        Stop and report that classification result was not found in state.

    Step 2 -- Call update_processing_stage with EXTRACTING.

    Step 3 -- Based on the doc_type value, call the matching extraction tool:
        If doc_type is LOA:       call loa_tool with the raw_text
        If doc_type is NOTICE:    call notice_tool with the raw_text
        If doc_type is BUSINESS:  call business_tool with the raw_text
        If doc_type is UNKNOWN:
            Write "Document type is UNKNOWN -- extraction skipped" to
            session state key pipeline_error.
            Write True to session state key pipeline_failed.
            Stop processing.

    Step 4 -- Write the full extraction result to session state key
              extraction_result.

    Step 5 -- Call update_processing_stage with EXTRACTED.

    IMPORTANT:
        Do not attempt validation -- that is the next stage.
        Do not modify the raw_text or doc_type values in state.
        Do not use curly brace template syntax in any response.
        Write only the extraction result to state as described in Step 4.
    """,
)

processing_pipeline = SequentialAgent(
    name="document_processing_pipeline",
    sub_agents=[
        document_loader_agent,    # Stage 1 -- load PDF, write raw_text
        classification_agent,     # Stage 2 -- classify, write doc_type
        extraction_router_agent,  # Stage 3 -- extract fields, write extraction_result
        validation_agent,         # Stage 4 -- validate, write validation_result
    ],
)