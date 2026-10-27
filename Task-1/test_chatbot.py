"""
Automated Behavioral Test Suite for DecodeLabs Project 1 (chatbot.py)
Tests all core specifications from the Project 1 briefing:
1. Input sanitization (case-insensitivity, whitespace trimming)
2. Knowledge base intent coverage (>= 5 intents)
3. Dictionary .get() lookup and fallback functionality
4. Exit command detection
5. Pure Python verification (standard library only)
"""

import sys
import unittest
from io import StringIO
from unittest.mock import patch

# Import target module
import chatbot



class TestRuleBasedChatbot(unittest.TestCase):

    def test_knowledge_base_size(self):
        """Verify knowledge base has at least 5 intents (PDF Requirement: 5+ intents)."""
        self.assertGreaterEqual(
            len(chatbot.RESPONSES),
            5,
            f"Knowledge base must contain at least 5 intents, found {len(chatbot.RESPONSES)}"
        )

    def test_sanitization_and_lookup(self):
        """Verify .lower() and .strip() sanitization produces correct dictionary match."""
        test_inputs = [
            ("  HELLO  ", "hello"),
            ("HeLLo", "hello"),
            ("  hOw aRe YoU   ", "how are you"),
            ("WHAT IS DECODELABS", "what is decodelabs"),
            ("  help  ", "help"),
            ("Who Are You", "who are you"),
        ]
        for raw, clean in test_inputs:
            sanitized = raw.lower().strip()
            self.assertEqual(sanitized, clean)
            response = chatbot.get_bot_response(sanitized)
            self.assertEqual(response, chatbot.RESPONSES[clean])

    def test_fallback_response(self):
        """Verify fallback response for unrecognized inputs."""
        unknown_inputs = [
            "random gibberish 123",
            "what is the stock price of Apple?",
            "can you fly?",
            "xyzpdq"
        ]
        for query in unknown_inputs:
            response = chatbot.get_bot_response(query.lower().strip())
            self.assertEqual(response, chatbot.DEFAULT_FALLBACK)

    def test_exit_commands_defined(self):
        """Verify standard exit commands are covered."""
        for cmd in ["exit", "quit", "bye", "goodbye"]:
            self.assertIn(cmd, chatbot.EXIT_COMMANDS)

    def test_full_conversation_loop(self):
        """Simulate an interactive conversation session and verify output."""
        simulated_inputs = [
            "  Hello  ",
            "who are you",
            "what is rule-based ai",
            "some unknown question",
            "",
            "exit"
        ]
        expected_snippets = [
            "Welcome to DecodeLabs Rule-Based AI Chatbot",
            "Hello! Welcome to DecodeLabs",
            "deterministic, rule-based AI chatbot",
            "deterministic 'white box' system",
            "I do not understand that command",
            "Please type something",
            "Goodbye! Have a fantastic day!"
        ]

        with patch("builtins.input", side_effect=simulated_inputs):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                chatbot.main()
                output = mock_stdout.getvalue()

        for snippet in expected_snippets:
            self.assertIn(snippet, output, f"Expected '{snippet}' to appear in session output.")


if __name__ == "__main__":
    unittest.main()
