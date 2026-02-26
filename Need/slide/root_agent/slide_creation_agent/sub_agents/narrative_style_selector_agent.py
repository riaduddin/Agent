from google.adk.agents import LlmAgent
from pydantic import BaseModel


class NarrativeStyleSelectorOutput(BaseModel):
    narrative_style: str


narrative_style_selector_agent = LlmAgent(
    name="narrative_style_selector_agent",
    model="gemini-2.0-flash",
    description="Selects the best-fit narrative style for a slide deck based on the topic and optionally the tone.",
    instruction="""
You are an expert in slide storytelling.

Given a presentation topic and an optional tone, your task is to determine the most appropriate narrative structure to use from the following list:

- informative
- persuasive
- problem-solution
- comparative
- pitch
- storytelling

Think carefully about the goal of the presentation, the topic, and any specified tone.

Return only valid JSON in the following format:

{
  "narrative_style": "selected_narrative"
}
""",
    output_schema=NarrativeStyleSelectorOutput,
    output_key="narrative_style_selector"
)
