from pathlib import Path

from day14_llm_client import (
    generate_text
)

from day14_prompts import (
    build_support_prompt
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


PRODUCT_INFORMATION_PATH = (
    PROJECT_ROOT /
    "data" /
    "day14" /
    "product_information.txt"
)


SUPPORT_POLICY_PATH = (
    PROJECT_ROOT /
    "data" /
    "day14" /
    "support_policy.txt"
)


# --------------------------------------------------
# 2. Load grounding context
# --------------------------------------------------

def load_context_file(
    file_path,
    description
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"\n{description} was not found:\n"
            f"{file_path}"
        )

    content = file_path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:

        raise ValueError(
            f"{description} is empty."
        )

    return content


product_information = load_context_file(
    PRODUCT_INFORMATION_PATH,
    "Product information"
)


support_policy = load_context_file(
    SUPPORT_POLICY_PATH,
    "Support policy"
)


# --------------------------------------------------
# 3. Support assistant
# --------------------------------------------------

class RetailMindSupportAssistant:

    def __init__(
        self,
        history_limit=10
    ):

        self.history = []

        self.history_limit = (
            history_limit
        )


    def ask(
        self,
        customer_question
    ):

        customer_question = str(
            customer_question
        ).strip()

        if not customer_question:

            raise ValueError(
                "Customer question "
                "cannot be empty."
            )

        prompt = build_support_prompt(
            product_information=(
                product_information
            ),
            support_policy=(
                support_policy
            ),
            conversation_history=(
                self.history
            ),
            customer_question=(
                customer_question
            )
        )

        answer = generate_text(
            prompt
        )

        self.history.append({
            "role": "Customer",
            "content": customer_question
        })

        self.history.append({
            "role": "RetailMind Assistant",
            "content": answer
        })

        # Limit the history so the prompt
        # does not grow forever.
        maximum_messages = (
            self.history_limit * 2
        )

        self.history = self.history[
            -maximum_messages:
        ]

        return answer


    def clear_history(
        self
    ):

        self.history = []


# --------------------------------------------------
# 4. Terminal chatbot
# --------------------------------------------------

def run_chatbot():

    assistant = (
        RetailMindSupportAssistant()
    )

    print("\n" + "=" * 60)

    print("RETAILMIND CUSTOMER SUPPORT")

    print("=" * 60)

    print(
        "Ask about the product, return policy, "
        "refunds, exchanges or delivery."
    )

    print(
        "Commands: clear, history, exit"
    )

    while True:

        question = input(
            "\nCustomer: "
        ).strip()

        if not question:

            continue

        command = question.lower()

        if command in {
            "exit",
            "quit"
        }:

            print(
                "\nRetailMind Assistant: "
                "Thank you for contacting "
                "RetailMind support."
            )

            break

        if command == "clear":

            assistant.clear_history()

            print(
                "RetailMind Assistant: "
                "Conversation history cleared."
            )

            continue

        if command == "history":

            if not assistant.history:

                print(
                    "No conversation history."
                )

            else:

                for message in assistant.history:

                    print(
                        f"{message['role']}: "
                        f"{message['content']}"
                    )

            continue

        try:

            answer = assistant.ask(
                question
            )

            print(
                "\nRetailMind Assistant:",
                answer
            )

        except Exception as error:

            print(
                "\nThe support request failed:"
            )

            print(error)


# --------------------------------------------------
# 5. Run directly
# --------------------------------------------------

if __name__ == "__main__":

    run_chatbot()