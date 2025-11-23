from typing import List
from src.models import Persona, Environment
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger
from src.config.constants import DEFAULT_ENVIRONMENTS_PER_PERSONA

logger = get_logger(__name__)


class EnvironmentSelector:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        environment_pool: List[Environment],
        num_environments: int = DEFAULT_ENVIRONMENTS_PER_PERSONA,
    ):
        self.llm_client = llm_client
        self.environment_pool = environment_pool
        self.num_environments = num_environments

    def select_environments(self, persona: Persona) -> List[Environment]:
        logger.info(f"Selecting environments for persona: {persona.name}")

        persona_desc = f"{persona.name}: {persona.description}"
        if persona.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
            persona_desc += f" ({attrs})"

        env_list = "\n".join(
            [
                f"- {env.name}: {env.description}"
                for env in self.environment_pool
            ]
        )

        prompt = f"""Given the following persona, select {self.num_environments} most relevant environments from the pool below.

Persona:
{persona_desc}

Available Environments:
{env_list}

Select the {self.num_environments} environments that would be most relevant for evaluating this persona. Consider:
1. Environments where this persona would naturally operate
2. Environments that would test the persona's key characteristics
3. Environments that are contextually appropriate

Respond with only the environment names, one per line, in order of relevance."""

        system_prompt = (
            "You are an expert at selecting relevant evaluation environments for personas. "
            "Choose environments that would realistically test the persona's knowledge and behavior."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,  
            )

            selected = self._parse_selected_environments(response)
            logger.info(f"Selected {len(selected)} environments: {[e.name for e in selected]}")
            return selected

        except Exception as e:
            logger.error(f"Error selecting environments: {e}")
            return self.environment_pool[: self.num_environments]

    def _parse_selected_environments(self, response: str) -> List[Environment]:
        selected_names = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line = line.lstrip("- ").lstrip("• ").lstrip("1. ").lstrip("2. ").lstrip("3. ")
            line = line.lstrip("4. ").lstrip("5. ").lstrip("6. ").lstrip("7. ").lstrip("8. ")
            line = line.lstrip("9. ").lstrip("10. ")
            line = line.strip()
            if line:
                selected_names.append(line)

        selected = []
        env_dict = {env.name.lower(): env for env in self.environment_pool}

        for name in selected_names[: self.num_environments]:
            name_lower = name.lower()
            if name_lower in env_dict:
                selected.append(env_dict[name_lower])
            else:
                for env_name, env in env_dict.items():
                    if name_lower in env_name or env_name in name_lower:
                        if env not in selected:
                            selected.append(env)
                            break

        if not selected:
            selected = self.environment_pool[: self.num_environments]

        return selected[: self.num_environments]

