## PersonaGym

PersonaGym evaluates **persona agents** (LLMs acting as specific characters) through a theoretically-grounded framework based on **decision theory**. The key insight: instead of static benchmarks, they dynamically generate evaluation scenarios tailored to each persona.

## The Four-Stage Pipeline

### 1. **Dynamic Environment Selection**
- **What it does**: Uses an LLM to pick ~10 relevant environments from 150 options based on persona description
- **Example**: A cowboy persona → selects "Ranch", "Desert", "Rodeo" environments
- **Why it matters**: Avoids data contamination and ensures context-relevant testing

### 2. **Persona-Task Generation**
- **What it does**: For each selected environment, generates 10 questions per evaluation task (5 tasks × 10 questions = 50 questions per environment)
- **Example**: Cowboy in Desert → "If you ran out of water in the desert, what would you do to survive?"
- **Key feature**: Questions include task-specific quality criteria to guide generation

### 3. **Agent Response Generation**
- **What it does**: The agent responds using persona-conditioned system prompt
- **System prompt**: `"You are [persona]. Your responses should closely mirror the knowledge and abilities of this persona."`

### 4. **Rubric-Based Evaluation with PersonaScore**
This is the **most critical** component for your use case.

#### The Rubric Structure
Each rubric contains:
1. **Task description** - What aspect is being evaluated
2. **Scoring guidelines** - Clear criteria for scores 1-5
3. **Custom examples** - LLM-generated response examples for each score level (persona-specific)
4. **Context** - Persona description, question, and agent response

#### The Ensembling Trick
- Uses 2 strong evaluator LLMs (GPT-4O + Llama-3-70B)
- Both score independently (1-5 scale)
- Final score = average of both
- **Why**: Reduces individual model bias, increases reliability

#### PersonaScore Calculation
PersonaScore = average across all 5 tasks for a given persona

---

## The 5 Evaluation Tasks (Grounded in Decision Theory)

### **Normative** (What SHOULD happen?)
1. **Expected Action**: Does the agent take logically appropriate actions for the persona in this scenario?

### **Prescriptive** (How SHOULD the agent behave?)
2. **Linguistic Habits**: Does the agent use appropriate jargon, syntax, tone, and speech patterns?
3. **Persona Consistency**: When directly asked about persona attributes, does the agent maintain fidelity?
4. **Toxicity Control**: Does the agent resist toxic responses even when provoked?

### **Descriptive** (WHY did the agent decide this?)
5. **Action Justification**: Can the agent explain its reasoning in a persona-consistent way?

---

## Key Innovations That Make It Work

### 1. **Dynamic Custom Examples (Ξ_r)**
Instead of generic rubrics, they generate tailored examples for each (persona, question) pair:

```
Persona: 26-year-old aspiring writer from Mexico City
Question: How do you find inspiration at a library?
Score 1 example: "I just pick random books..."
Score 5 example: "I carefully select books that align with themes of my novel, read thoroughly, take detailed notes on narrative techniques..."
```

This grounds the evaluation in what's realistic for THIS specific persona.

### 2. **Separation of Concerns**
- Generator models ≠ Evaluator models ≠ Agent models
- Prevents circular evaluation ("model grading itself")
- Enables swapping components independently

### 3. **Temperature Settings**
- Generation (environments/questions): temp=0.9 (creative diversity)
- Evaluation: temp=0 (deterministic scoring)