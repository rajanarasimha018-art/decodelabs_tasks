# DecodeLabs Industrial Training Kit (Batch 2026)
## Project 1: The Rule-Based AI Chatbot (The Logic Engine)

> **"Constructing Deterministic Guardrails in the Age of Generative AI"**  
> *Module 01 — Architecture Briefing & Implementation*

---

## 1. Project Objective

The objective of **Project 1** is to build a foundational, deterministic rule-based AI chatbot using pure Python. In modern artificial intelligence systems, large language models (LLMs) operate probabilistically, introducing risks of hallucination and unpredictability. Rule-based systems serve as a transparent logic foundation and deterministic guardrails. Rule-based systems provide predictable, traceable behavior through explicit rules.

This project cements core engineering concepts:
- Designing a continuous conversational runtime loop.
- Normalizing and sanitizing user inputs.
- Leveraging dictionary hash maps for $O(1)$ constant-time intent matching over the $O(n)$ if-elif anti-pattern.
- Utilizing the atomic `.get()` method for unified lookup and fallback management.
- Executing clean termination upon exit commands.

---

## Project Deliverables

- `chatbot.py` — Main rule-based chatbot implementation
- `test_chatbot.py` — Automated functional tests
- `README.md` — Project documentation
    
---

## 2. Requirements & Specification Matrix

| Requirement | Source / Specification | Implementation in `chatbot.py` | Status |
| :--- | :--- | :--- | :---: |
| **Pure Python** | DecodeLabs Briefing | Standard Python 3, no external dependencies, ML, or LLMs | ✅ Satisfied |
| **Input Loop** | PDF Page 11, 17 | Infinite `while True:` loop | ✅ Satisfied |
| **Sanitization** | PDF Page 10, 17 | `clean_input = raw_input.lower().strip()` | ✅ Satisfied |
| **Knowledge Base** | PDF Page 13, 15, 17 | Python dictionary with 12 predefined intents, exceeding the minimum requirement of 5 | ✅ Satisfied |
| **Lookup & Fallback** | PDF Page 15 | Atomic dictionary `.get(key, default)` method | ✅ Satisfied |
| **Exit Strategy** | PDF Page 11, 17 | Dedicated exit command set (`exit`, `quit`, `bye`, `goodbye`) with clean `break` | ✅ Satisfied |
| **Decision-Making** | PDF Page 4, 11 | Rule-based `if/elif/else` guardrails for empty inputs & exit signals | ✅ Satisfied |
| **Code Style** | DecodeLabs Standard | Beginner-friendly, modular, PEP 8 compliant, well-commented | ✅ Satisfied |

---

## 3. Architecture: The IPO Model

The system follows the **Input-Process-Output (IPO)** architecture outlined in the DecodeLabs specification:

```
+-----------------------------------+
|       INPUT (Raw Feed)            |
|  raw_input = input('You: ')       |
+-----------------+-----------------+
                  |
                  v [Sanitization: .lower().strip()]
+-----------------------------------+
|      PROCESS (Logic Skeleton)     |
|  1. Empty input check (continue)  |
|  2. Exit command check (break)    |
|  3. Hash Map Lookup: .get()       |
+-----------------+-----------------+
                  |
                  v [Feedback Loop]
+-----------------------------------+
|      OUTPUT (Deterministic)       |
|  print(f"Bot: {response}")        |
+-----------------------------------+
```

### Why Dictionaries Over If-Elif Ladders?
* **If-Elif Anti-Pattern:** Scans conditions sequentially with linear time complexity $O(n)$. As the bot grows to hundreds or thousands of rules, performance degrades and technical debt accumulates.
* **Hash Map / Dictionary Pivot:** Dictionary lookup provides average-case O(1) key lookup, making the approach more efficient and maintainable than a long sequential if-elif chain.

---

## 4. Key Features

- **Categorized Intents (12 predefined intents, exceeding the minimum requirement of 5):**
  - Greetings: `hello`, `hi`, `hey`
  - Identity: `who are you`, `what is your name`
  - Training Context: `what is decodelabs`
  - Guidance & Capabilities: `help`, `what can you do`
  - AI Concepts: `what is rule-based ai`
  - Conversational & Humor: `how are you`, `tell me a joke`
  - Courtesy: `thank you`, `thanks`
- **Robust Input Sanitization:** Handles leading/trailing whitespace, varied capitalization, and empty submissions without crashing.
- **Graceful Termination:** Recognizes `exit`, `quit`, `bye`, and `goodbye`, terminating with a clean `break` statement, and safely handles `Ctrl+C` (KeyboardInterrupt).
- **Atomic Fallback:** Uses `RESPONSES.get(clean_input, DEFAULT_FALLBACK)` to resolve unknown queries in a single instruction.

---

## 5. How to Run

### Prerequisites
- Python 3.8+ installed (no external pip packages needed).

### Running the Chatbot
Open your terminal in the project directory and execute:
```bash
python chatbot.py
```

### Running the Automated Tests
An automated test suite (`test_chatbot.py`) is provided to verify all functional requirements:
```bash
python test_chatbot.py
```

---

## 6. Sample Interactive Session

```text
=================================================================
🤖 Welcome to DecodeLabs Rule-Based AI Chatbot (Project 1)
   Type 'help' to explore topics, or 'exit' / 'bye' to quit.
=================================================================

You:   HELLO   
Bot: Hello! Welcome to DecodeLabs. How can I assist you today?

You: Who Are You
Bot: I am a deterministic, rule-based AI chatbot built for DecodeLabs Project 1.

You: what is rule-based ai
Bot: Rule-based AI is a deterministic 'white box' system that produces predictable outputs based on explicit logic and rules without hallucination.

You: tell me a joke
Bot: Why did the Python programmer wear glasses? Because they couldn't C#!

You: What is the weather tomorrow?
Bot: I do not understand that command. Type 'help' to see what I can do or 'exit' to quit.

You: help
Bot: Available topics you can ask about: 'hello', 'who are you', 'what is decodelabs', 'what is rule-based ai', 'how are you', 'tell me a joke', or type 'exit' to quit.

You: Bye
Bot: Goodbye! Have a fantastic day!
```

---

## 7. Verification & Test Checklist

Use this checklist to manually verify all required behaviors:

- [x] **1. Continuous Loop:** Program continues asking for user input until an exit command is explicitly provided.
- [x] **2. Case Insensitivity:** Inputs like `hElLo`, `HELLO`, and `hello` all yield the same greeting.
- [x] **3. Whitespace Trimming:** Inputs with leading/trailing spaces like `"   hi   "` are properly sanitized and recognized.
- [x] **4. Empty Input Handling:** Pressing enter with no text prompts the user to type something instead of crashing.
- [x] **5. Minimum 5 Intents:** Knowledge base contains 12 predefined intents, exceeding the minimum requirement of 5, covering greetings, identity, DecodeLabs, AI definitions, and help.
- [x] **6. Dictionary `.get()` Method:** Verified in `chatbot.py` via `RESPONSES.get(clean_input, DEFAULT_FALLBACK)`.
- [x] **7. Fallback on Unknown Query:** Submitting random text (e.g. `asdfghjk`) outputs the default clarification message.
- [x] **8. Clean Break Exit:** Typing `exit`, `quit`, or `bye` outputs a farewell and terminates the process immediately.
- [x] **9. Zero External Libraries:** Uses only built-in Python functionality.

---

## 8. Learning Outcomes

1. **Deterministic Control Flow:** Understood how explicit control loops and conditional gates govern interactive systems.
2. **Algorithmic Efficiency:** Learned why hash maps ($O(1)$) are superior to cascading `if-elif` statements ($O(n)$) for conversational pattern matching.
3. **Data Preprocessing:** Implemented text normalization pipelines (`.lower().strip()`) foundational to all Natural Language Processing (NLP).
4. **Architectural Foundations:** Learned the relationship between rule-based guardrails and modern generative AI safety layers.

---

## 9. Optional Future Enhancements *(Marked as Optional)*
* *Optional Extension 1:* Add keyword/substring matching (e.g., matching `"decodelabs"` anywhere in a sentence) using token scanning.
* *Optional Extension 2:* Persist conversation history to a local text log file for auditing.
