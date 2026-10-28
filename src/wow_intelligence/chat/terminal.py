from importlib.metadata import version
import sys

from wow_intelligence.chat.policies.conversation import ConversationPolicy
from wow_intelligence.chat.policies.domain import WoWDomainPolicy
from wow_intelligence.chat.policies.language import EnglishLanguagePolicy
from wow_intelligence.chat.prompts import AssistantInstructions
from wow_intelligence.config.ollama import OllamaConfig
from wow_intelligence.ollama.client import generate
from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.ollama.messages import Message
from wow_intelligence.rag.generation.catalog_answers import catalog_names
from wow_intelligence.rag.generation.prompts import ABSTENTION
from wow_intelligence.rag.models.rag_answer import RagAnswer
from wow_intelligence.rag.pipeline import RagPipeline


def run_chat(config: OllamaConfig | None = None, *, rag: RagPipeline | None = None) -> None:
    config = config if config is not None else OllamaConfig()
    history: list[Message] = [AssistantInstructions().system_message()]
    language_policy = EnglishLanguagePolicy()
    domain_policy = WoWDomainPolicy()
    last_rag_answer = RagAnswer(ABSTENTION)
    rag_topic_available = False
    print("WoW Intelligence — experimental terminal chat")
    print("Conversation context lasts for this session only; nothing is saved to disk.")
    if rag is None:
        print("Context is limited. Answers are not verified against sources.")
    else:
        print(f"RAG enabled for Forever build {rag.game_build}. Context is limited.")
        if rag.include_catalog:
            print("Source catalog mode: beta data, not independently verified in game.")
        print("Citations identify retrieved evidence; generated claims still need review.")
    print("Type /help for commands, /exit to leave, or press Ctrl+C.")
    print("English only. Ambiguous input may require a complete English sentence.")

    try:
        while True:
            prompt = input("You: ").strip()
            if prompt == "/exit":
                break
            if not prompt:
                continue
            if prompt == "/help":
                print("Ask questions in English about World of Warcraft: Forever.")
                print("/help — Show this help.")
                print("/version — Show the application version.")
                print("/exit — End the session and discard its history.")
                print("/clear — Clear the conversation and start a new topic.")
                if rag is not None:
                    print("/sources — Show citations from the last accepted answer.")
                    print("/coverage — Show loaded evidence and build coverage.")
                print("History stays in memory for this session only. Context is limited.")
                if rag is None:
                    print("Answers are not verified against sources.")
                else:
                    print("Name the subject of your question. Follow-ups can use the previous topic.")
                continue
            if prompt == "/clear":
                history = [AssistantInstructions().system_message()]
                last_rag_answer = RagAnswer(ABSTENTION)
                rag_topic_available = False
                print("Conversation cleared.")
                continue
            if prompt == "/version":
                print(f"WoW Intelligence {version('wow-intelligence')}")
                continue
            if prompt == "/sources" and rag is not None:
                print(last_rag_answer.sources_text())
                continue
            if prompt == "/coverage" and rag is not None:
                print(rag.coverage_text())
                continue
            if prompt.startswith("/"):
                print("Unknown command. Type /help for available commands.")
                continue
            if not language_policy.is_english(prompt, **({"known_names": rag.language_names} if rag else {})):
                rag_topic_available = False
                print(language_policy.input_rejection)
                continue
            if rag is not None:
                local_reply = ConversationPolicy.local_reply(prompt)
                if local_reply is not None:
                    last_rag_answer = RagAnswer(local_reply)
                    print(f"Assistant: {local_reply}")
                    continue
                if ConversationPolicy.needs_context(prompt) and not rag_topic_available:
                    print("Assistant: Which item, quest, class, or location do you mean?")
                    last_rag_answer = RagAnswer(ABSTENTION)
                    continue
            previous_question = next(
                (message["content"] for message in reversed(history) if message["role"] == "user"),
                "",
            )
            if domain_policy.is_clearly_unrelated(prompt, previous_question):
                rag_topic_available = False
                print(domain_policy.input_rejection)
                continue
            messages: list[Message] = [*history, {"role": "user", "content": prompt}]
            try:
                prepared = rag.prepare(history, prompt) if rag is not None else None
                if prepared is not None and not prepared.evidence:
                    rag_topic_available = False
                    last_rag_answer = RagAnswer(prepared.notice or ABSTENTION)
                    print(f"Assistant: {last_rag_answer.text}")
                    continue
                if rag is not None:
                    def configured_generate(request, *, response_schema):
                        return generate(request, model=config.model, base_url=config.base_url,
                                        timeout=config.timeout, response_schema=response_schema)
                    rag_answer = rag.answer(prepared, prompt, generator=configured_generate)
                    answer = rag_answer.text
                else:
                    rag_answer = None
                    answer = generate(messages, model=config.model, base_url=config.base_url, timeout=config.timeout)
            except OllamaError as exc:
                rag_topic_available = False
                print(f"Error: {exc}", file=sys.stderr)
                continue
            validation_text = (rag_answer.validation_text or answer) if rag_answer else answer
            names = catalog_names(rag_answer.evidence) if rag_answer else ()
            if not language_policy.is_english(validation_text, **({"known_names": names} if names else {})):
                rag_topic_available = False
                print(language_policy.output_rejection)
                continue
            history = [*messages, {"role": "assistant", "content": answer}]
            print(f"Assistant: {answer}")
            if rag_answer is not None:
                rag_topic_available = bool(rag_answer.evidence)
                last_rag_answer = rag_answer
                if rag_answer.evidence:
                    print(rag_answer.sources_text())
    except (EOFError, KeyboardInterrupt):
        print()

    print("Goodbye.")
