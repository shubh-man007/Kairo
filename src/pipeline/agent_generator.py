from src.models import Persona, Question, AgentResponse
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger
from src.config import get_settings

logger = get_logger(__name__)


class AgentGenerator:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.settings = get_settings()

    def generate_response(
        self,
        persona: Persona,
        question: Question,
    ) -> AgentResponse:
    
        logger.info(
            f"Generating response for persona={persona.name}, "
            f"question={question.text[:50]}..."
        )

        system_prompt = self._build_system_prompt(persona)

        try:
            response_text = self.llm_client.generate(
                prompt=question.text,
                system_prompt=system_prompt,
                temperature=self.settings.generator_temperature,
            )

            response = AgentResponse(
                text=response_text,
                persona_name=persona.name,
                metadata={
                    "model": self.llm_client.model,
                    "temperature": self.settings.generator_temperature,
                    "task_type": question.task_type,
                    "environment": question.environment_name,
                },
            )

            logger.info(f"Generated response ({len(response_text)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return AgentResponse(
                text=f"Error generating response: {str(e)}",
                persona_name=persona.name,
                metadata={"error": str(e)},
            )

    def _build_system_prompt(self, persona: Persona) -> str:
        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

        system_prompt = f"""You are {persona_desc}

Your responses should closely mirror the knowledge, abilities, linguistic patterns, and decision-making processes of this persona. Act as if you are this persona, not an AI assistant. Do not acknowledge that you are an AI or language model. Respond naturally and consistently with the persona's characteristics."""

        return system_prompt

