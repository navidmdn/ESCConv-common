from typing import List, Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

strategy_descriptions = {
    "Question": """In emotional support counseling, asking open-ended questions is crucial. These questions encourage the\
 individual to express their thoughts and feelings. They can be used to explore the person's emotions, triggers,\
 or concerns. For instance, you might ask, "What is causing you to feel this way?" """,
    "Restatement or Paraphrasing": """This technique involves repeating or rephrasing what the person has said to show\
 that you're actively listening and trying to understand. For instance, if someone says, "I'm really stressed about\
 work," you can respond with, "It sounds like work is causing you a lot of stress." """,
    "Reflection of Feelings": """Reflecting feelings involves acknowledging and validating the person's emotions. This\
 helps the individual feel heard and understood. If someone expresses sadness, you might respond with, "I can see that\
 you're feeling very sad about this situation." """,
    "Self-disclosure": """Carefully sharing your own relevant experiences or feelings can help build rapport and trust.\
 However, it should be done in a way that doesn't shift the focus from the person seeking support. For example,\
 "I've also felt overwhelmed at times, and it's completely normal." """,
    "Affirmation and Reassurance": """Providing affirmations and reassurance helps boost the individual's self-esteem and\
 confidence. You might say, "You're doing a great job by seeking help, and I believe you have the strength to overcome\
 this." """,
    "Providing Suggestions": """In emotional support counseling, suggesting helpful actions or strategies can guide\
 individuals without imposing decisions. For example, if someone is stressed, you might suggest trying relaxation\
 techniques like deep breathing.""",
    "Information": """Providing relevant facts or resources is crucial in emotional support. Offering accurate information\
 about available therapies, self-help options, or support groups can empower individuals to make informed decisions\
 about their emotional well-being."""
}