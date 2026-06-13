from google.adk.agents import LlmAgent, LoopAgent
from google.adk.tools.agent_tool import AgentTool

from ..util.settings import MODEL_FLASH_LITE

from .loa_extraction_agent    import loa_extraction_agent
from .notice_extraction_agent import notice_extraction_agent
from .business_doc_agent      import business_doc_agent
from .validation_agent        import validation_agent

qr_loa_tool        = AgentTool(agent=loa_extraction_agent)
qr_notice_tool     = AgentTool(agent=notice_extraction_agent)
qr_business_tool   = AgentTool(agent=business_doc_agent)
qr_validation_tool = AgentTool(agent=validation_agent)

quality_check_agent = LlmAgent(
    name="quality_check_agent",
    model=MODEL_FLASH_LITE,
    tools=[
        qr_loa_tool,
        qr_notice_tool,
        qr_business_tool,
        qr_validation_tool,
    ],
    instruction="""You are the quality refinement agent.
    Your job is to check the current extraction quality and retry
    extraction with targeted feedback if the confidence is too low.

    You run inside a loop. On each iteration you check whether the
    extraction has reached the required quality threshold. If it has,
    you signal the loop to stop. If not, you try to improve it.

    QUALITY CHECK STEPS:

    Step 1 -- Read the current state values:
        Read validation_result from session state key validation_result.
        Read extraction_result from session state key extraction_result.
        Read doc_type from session state key doc_type.
        Read raw_text from session state key raw_text.

    Step 2 -- Check the confidence_score from validation_result.
        If validation_result is not available in state:
            Write "No validation result found in state" to session state
            key refinement_error.
            Signal escalate_to_parent to stop the loop.
            Stop here.

    Step 3 -- If confidence_score is 0.90 or above:
        The extraction quality is sufficient.
        Write "Quality threshold met" to session state key refinement_status.
        Signal escalate_to_parent to stop the loop.
        Stop here -- do not retry.

    Step 4 -- If confidence_score is below 0.90:
        Read the missing_fields list from validation_result.
        Read the recommendations list from validation_result.
        Read the review_reason from validation_result.

        Build targeted feedback by examining the missing fields.
        For each missing field, think about where in the document
        that information typically appears and how it might be phrased.

        Example feedback for common missing fields:
            For authorizing_party: look for the name before "hereby authorizes"
            or after "I the undersigned" or in the signature block.
            For authorized_party: look for the name after "authorize" or
            after "on behalf of" or before "to act as".
            For effective_date: look for dates near "effective", "commencing",
            "starting", "from the date of", or at the document header.
            For notice_type: look at the document title or first sentence.
            For parties_involved: look at the opening paragraph listing parties.
            For reference_numbers: look in headers, footers, or the subject line.

    Step 5 -- Re-run the correct extraction tool with the raw_text.
        Pass the raw document text to give the extraction agent
        another chance to find the missing fields.
        Based on the doc_type value:
            If doc_type is LOA:      call qr_loa_tool with raw_text
            If doc_type is NOTICE:   call qr_notice_tool with raw_text
            If doc_type is BUSINESS: call qr_business_tool with raw_text
            If doc_type is UNKNOWN:
                Write "Cannot refine UNKNOWN document type" to session state
                key refinement_error.
                Signal escalate_to_parent.
                Stop here.

    Step 6 -- Update session state with the new extraction result:
        Write the new extraction result to session state key extraction_result.

    Step 7 -- Re-validate the new extraction:
        Call qr_validation_tool with the new extraction result.
        Write the new validation result to session state key validation_result.

    Step 8 -- Write the current iteration status to state:
        Write a summary of what was improved to session state key
        refinement_status. Include the new confidence_score value
        from the updated validation result.

    LOOP BEHAVIOUR:
        After Step 8, the loop will automatically run this agent again
        if max_iterations has not been reached.
        On the next iteration, Step 3 will check the new confidence_score.
        If it is now 0.90 or above, the loop will stop via escalate_to_parent.
        If it is still below 0.90 and iterations remain, another retry fires.
        When max_iterations is reached, the loop stops automatically.

    IMPORTANT:
        Do not use curly brace template syntax anywhere in your responses.
        Do not modify raw_text or doc_type state values during refinement.
        Always write updated results back to state after each retry.
        The escalate_to_parent signal stops the loop -- use it when done.
    """,
)

quality_refinement_loop = LoopAgent(
    name="quality_refinement_loop",
    sub_agents=[quality_check_agent],
    max_iterations=2,
)