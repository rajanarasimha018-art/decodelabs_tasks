"""
DecodeLabs Industrial Training Kit - Project 1: The Rule-Based AI Chatbot
Module: Logic Engine & Deterministic Guardrails
Architecture: IPO Model (Input -> Process -> Output)

Description:
A deterministic, rule-based chatbot implemented in pure Python.
Uses an infinite loop (while True), input sanitization (.lower().strip()),
if-else decision logic, and dictionary-based O(1) hash map retrieval
with the .get() method for atomic response lookup and fallback handling.
"""

# ==========================================
# 1. KNOWLEDGE BASE (Dictionary / Hash Map)
# ==========================================
# Rule-based mapping of user intents (keys) to predefined responses (values).
# Provides O(1) constant-time direct access lookup.
RESPONSES = {
    # Greetings
    "hello": "Hello! Welcome to DecodeLabs. How can I assist you today?",
    "hi": "Hi there! I am your rule-based AI assistant. Ask me anything from my knowledge base.",
    "hey": "Hey! Great to see you. How can I help you?",
    
    # Bot Identity & Info
    "who are you": "I am a deterministic, rule-based AI chatbot built for DecodeLabs Project 1.",
    "what is your name": "I am LogicBot, your DecodeLabs AI Guide.",
    
    # Project & Organization Context
    "what is decodelabs": "DecodeLabs is an industrial technology training platform empowering future AI engineers.",
    
    # Capabilities & Guidance
    "what can you do": "I can answer predefined queries about AI, DecodeLabs, and demonstrate rule-based control flow.",
    "help": "Available topics you can ask about: 'hello', 'who are you', 'what is decodelabs', 'what is rule-based ai', 'how are you', 'tell me a joke', or type 'exit' to quit.",
    
    # Core AI Knowledge
    "what is rule-based ai": "Rule-based AI is a deterministic 'white box' system that produces predictable outputs based on explicit logic and rules without hallucination.",
    
    # Conversational & Fun
    "how are you": "I am functioning within optimal operational parameters! How are you doing today?",
    "tell me a joke": "Why did the Python programmer wear glasses? Because they couldn't C#!",
    
    # Appreciation
    "thank you": "You're very welcome! Let me know if you need anything else.",
    "thanks": "Glad I could help!"
}

# Predefined set of exit commands
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

# Fallback response for unrecognized inputs
DEFAULT_FALLBACK = "I do not understand that command. Type 'help' to see what I can do or 'exit' to quit."


def get_bot_response(clean_input: str) -> str:
    """
    Lookup response in the knowledge base using the dictionary .get() method.
    Handles response matching and fallback atomically in O(1) time complexity.
    """
    return RESPONSES.get(clean_input, DEFAULT_FALLBACK)


def main():
    """
    Main conversation cycle implementing the IPO (Input-Process-Output) model:
    - Input: raw_input = input('You: ')
    - Sanitization: clean_input = raw_input.lower().strip()
    - Process: Rule-based decision checks (empty input, exit commands, knowledge base lookup)
    - Output: Deterministic bot reply displayed to the user
    """
    print("=" * 65)
    print("🤖 Welcome to DecodeLabs Rule-Based AI Chatbot (Project 1)")
    print("   Type 'help' to explore topics, or 'exit' / 'bye' to quit.")
    print("=" * 65)
    print()

    # The Heartbeat: Infinite loop that keeps the chatbot active
    while True:
        try:
            # Phase 1: Input & Sanitization
            raw_input = input("You: ")
            clean_input = raw_input.lower().strip()

            # Rule 1: Handle empty inputs (e.g., user just pressed enter)
            if not clean_input:
                print("Bot: Please type something so I can respond! (or type 'help')")
                continue

            # Rule 2: Handle exit strategy (clean break command)
            if clean_input in EXIT_COMMANDS:
                print("Bot: Goodbye! Have a fantastic day!")
                break

            # Phase 2 & 3: Process via Dictionary .get() & Output
            response = get_bot_response(clean_input)
            print(f"Bot: {response}")

        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C or end-of-file interrupt
            print("\nBot: Session interrupted. Exiting cleanly. Goodbye!")
            break


if __name__ == "__main__":
    main()
