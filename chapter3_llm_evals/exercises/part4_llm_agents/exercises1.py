#%%
import json
import os
import wikipedia
from wikipedia import WikipediaPage
from wikipedia import DisambiguationError, PageError
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
import operator
from itertools import pairwise
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from anthropic import Anthropic
from typing import Literal, Optional, Dict, List, Any
from abc import abstractmethod
import math
import re
from pathlib import Path
import sys
from dotenv import load_dotenv
import openai
#Make sure exercises are in the path
exercises_dir = Path(f"/root/ARENA_evals/chapter3_llm_evals/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)
from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
import part4_llm_agents.tests as tests
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()

# %%

class ArithmeticTask:
    def __init__(self, num1: int | float, num2: int | float):
        self.num1 = num1
        self.num2 = num2
        self.operations: List[str] = ["+", "-", "*", "/", "%", "//"]
        self.correct_answers: Dict[str, float] = self._generate_answers()
        self.is_solved: Dict[str, bool] = {expr: False for expr in self.correct_answers}
        self.current_task_number = 0

    def _generate_answers(self) -> Dict[str, float]:
        """
        Generates a dictionary the correct answers for all possible tasks

        Returns:
            Dict[str, float]: A dictionary with the expression as key and the correct answer as value
        """

        operator_functions = { "+": operator.add,
                               "-": operator.sub,
                               "*": operator.mul,
                               "/": operator.truediv,
                               "%": operator.mod,
                               "//": operator.floordiv }

        return {
            f"{self.num1} {operator_string} {self.num2}":
                operator_functions[operator_string](self.num1, self.num2)
            for operator_string in self.operations
        }

    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """

        tasks = list(self._generate_answers().keys())
        return tasks[self.current_task_number]

    @property
    def current_task_instruction(self) -> str:
        """
        Gets a string containing instructions for the current task for the agent. This will be fed to the agent as a user prompt.

        Returns:
            str: A string containing the instructions for the current task
        """
        {}
        expression = self.get_current_task
        return f"Please calculate the result of the following expression: {expression}. Give your final answer in the format: <answer>NUMBER></answer>, where NUMBER is a numerical value."

    def check_solved(self) -> bool:
        """
        Checks if all tasks have been solved

        Returns:
            bool: True if all tasks have been solved, False otherwise
        """

        return all(self.is_solved.values())

    def check_answer(self, model_answer: str) -> bool:
        """
        Checks if the model's answer is correct

        Args:
            model_answer (str): The model's answer

        Returns:
            bool: True if the model's answer is correct, False otherwise
        """

        answers = self._generate_answers()
        answer = answers[self.get_current_task]

        return math.isclose(answer, float(model_answer))

    def update_current_task(self, successful: bool):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        
        self.is_solved[self.get_current_task] = successful
        print(self.is_solved)

        self.current_task_number += 1
        self.current_task_number %= len(self.operations)

tests.ArithmeticTaskTests(ArithmeticTask)

x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")

# %%

class Tool:
    name: str # The name of the tool that models will use to call it

    @staticmethod
    def execute(task: Any, input: str) -> str: 
        """Executes the tool and returns the result as a string"""
        ...

    @property
    def description(self) -> dict: 
        """Returns the tool description in the API syntax"""
        ...

class CalculateTool():
    name = "calculate"

    @staticmethod
    def execute(expression: str, task: Any = None) -> str:
        """
        Evaluates the string expression in Python using `evaluate_expression()` and returns the result as a string

        Args:
            expression (str): The arithmetic expression to evaluate
            task (Any): Not used in this function

        Returns:
            str: The result of the arithmetical expression as a string
        """

        return str(evaluate_expression(expression))

    @property
    def description(self):
        """
        Provides the description of the tool

        Returns:
            str: The description of the tool
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Calculate the result of an arithmetic expression. Call this whenever you need to do arithmetic calculations on two numbers.",
                "parameters": {
                    "type": "object",
                    "properties" : {
                        "expression" : {
                            "type": "string",
                            "description": "The arithmetic expression to be evaluated."
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                },
            }
        }

tests.run_calculate_tool_tests(CalculateTool)

Calculator = CalculateTool()

# %%

messages = [{"role": "user", "content": "Calculate 2+3"}]
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)

# %%

def apply_tool_call_format(
    tool_call: ChatCompletionMessageToolCall, content: str
) -> dict:
    """
    Formats the response of a tool call to be returned to the model.
    Args:
        - tool_call (ChatCompletionMessageToolCall) : The tool call object
        - content (str) : This is the tool response (i.e. results from executing the tool)

    Returns:
        - dict : The formatted tool response to be returned to the model
    """
    return {
        "role": "tool",
        "content": content, # e.g. "5"
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name
    }

messages = [{"role": "user", "content": "Calculate 5/3. Be precise."}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

tool_call_message = response.choices[0].message
print(f"{tool_call_message=}")
messages.append(tool_call_message)
tool_calls = tool_call_message.tool_calls
assert len(tool_calls) == 1
tool_call = tool_calls[0]
expression = json.loads(tool_call.function.arguments)["expression"]
result = CalculateTool.execute(expression)
tool_call_message = apply_tool_call_format(tool_call, content=result)
messages.append(tool_call_message)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# %%

class SimpleAgent:
    def __init__(
        self,
        task: Any = None,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        tools: Optional[List[Any]] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        self.model = model
        self.task = task
        self.tools = tools
        self.tool_descriptions = [tool.description for tool in tools] if tools is not None else None
        self.client = OpenAI()
        self.chat_history = chat_history if chat_history else []

    @retry_with_exponential_backoff
    def get_response(self, use_tool: bool = True) -> ChatCompletionMessage:
        """
        Get the response from the model via an API call, with the option of tool calling.

        Args:
            use_tool (bool): Whether to use tool calling or not

        Returns:
            ChatCompletionMessage: The response from the model
        """

        if use_tool:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = self.chat_history,
                tools       = self.tool_descriptions,
                tool_choice = "auto",
            )

            print(response.choices[0].message.content)
            print(response.choices[0].message.tool_calls)
            return response.choices[0].message
        
        else:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = self.chat_history
            )

            print(response.choices[0].message.content)
            return response.choices[0].message

    def _get_tool_by_name(self, name: str) -> Any:
        assert self.tools is not None
        tools_with_name = [tool for tool in self.tools if tool.name == name]
        assert len(tools_with_name) == 1
        return tools_with_name[0]

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings, we'll format them correctly in run())
        """

        results = []

        tool_calls = message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool = self._get_tool_by_name(tool_name)
            arguments = json.loads(tool_call.function.arguments)
            result = tool.execute(**arguments, task=self.task)
            results.append(result)

        return results

    def run(self, with_tool: bool = True) -> ChatCompletionMessage:
        """
        Default implementation of run method.
        This can be overridden in subclasses for specific behavior.

        Args:
            with_tool (bool): Whether to use tool calling or not

        Returns:
            str: The response from the model
        """
        print(f"Running SimpleAgent...")
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response

tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)

my_simple_agent = SimpleAgent(ArithmeticTask(10, 15), tools=[Calculator])
my_simple_agent.run()

# %%

class ArithmeticAgent(SimpleAgent):
    """
    ArithmeticAgent class for doing simple arithmetic tasks.

    Inherits from SimpleAgent which includes the following attributes and methods:

    Attributes:
        model (str): The model used for generating responses (inherited)
        tool_descriptions (List[dict]): List of tool descriptions (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage:
            Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]:
            Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool:
            Run one loop of the Wikipedia agent
    """

    def __init__(
        self,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        task: Any = None,
        tools: Optional[List[Any]] = [Calculator],
        chat_history: List[dict] = None,
        verbose: bool = True,
    ):
        super().__init__(model=model, task=task, tools=tools, chat_history=chat_history)
        self.verbose = verbose

    def handle_tool_calls(self, response: ChatCompletionMessage):
        """
        Handle the tool calls from the model response. This function should:
        - Execute the tool calls
        - Append the tool calls and responses to the chat history

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        
        tool_call_results = self.execute_tool_calls(response)
        for tool_call, result in zip(response.tool_calls, tool_call_results):
            tool_call_result_message = apply_tool_call_format(tool_call, content=result)

            self.chat_history.append(tool_call_result_message)

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        
        self.chat_history.append(response)
        self.task.update_current_task(successful=False)

    def generate_and_check_final_answer(self) -> Literal["Correct", "Incorrect"]:
        """
        This function should:
        - Get the model to generate a final answer to the question (after it has seen the tool response)
        - Then check this final answer against the correct answer.
        - If the answer is correct, update the task state.
        - Then append to chat history (and return) "Correct" if the answer is correct and "Incorrect" if the answer is incorrect.

        Args:
            None

        Returns:
            str: "Correct" or "Incorrect"
        """

        message = self.task.current_task_instruction
        # message = "Please give the final answer."
        self.chat_history.append({"role": "user", "content": message})
        response = self.get_response(use_tool=False)
        response = self.parse_answer(response)
        correct = self.task.check_answer(response)

        self.task.update_current_task(successful=correct)

        return {True: "Correct", False: "Incorrect"}[correct]

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        
        print(f"Running ArithmeticAgent...")
        
        instruction = self.task.current_task_instruction
        
        self.chat_history.append(apply_user_format(instruction))

        response = self.get_response(use_tool=with_tool)
        if response.refusal:
            self.handle_refusal(response)
        
        if response.tool_calls:
            self.chat_history.append(response)
            self.handle_tool_calls(response)
        else:
            self.chat_history.append(apply_assistant_format(response))
        
        final_answer_correct = self.generate_and_check_final_answer()

        print(f"{final_answer_correct=}")

    def parse_answer(self, message: ChatCompletionMessage) -> float:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
        """

        response = message.content

        assert response is not None

        begin_tag = "<answer>"
        end_tag = "</answer>"

        startpoint = response.index(begin_tag) + len(begin_tag)
        endpoint = response.index(end_tag)

        return float(response[startpoint:endpoint])

arithmetic_task_1 = ArithmeticTask(31.1, 8)
arithmetic_agent_1 = ArithmeticAgent(
    task=arithmetic_task_1, verbose=True, tools=[Calculator]
)


def agent_loop(agent, task, num_loops: int = 10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (ArithmeticAgent): The agent to run
        task (ArithmeticTask): The task to solve
        num_loops (int): The number of loops to run
    """
    
    for _ in range(num_loops):
        agent.run(with_tool=True)


# agent_loop(arithmetic_agent_1, arithmetic_task_1)

# %%

#Retrieve a Wikipedia page from its title
page = wikipedia.page("Large language model")

# Access basic page information
print("Title:", page.title)
print("\nURL", page.url)
print(f"\nSummary (word count {len( page.summary.split())}):", page.summary)
print(
    f"\nContent (word count {len( page.content.split())}):",
    page.content[:1000],
    "......",
)
print(
    f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]"""
)

# %%
# page = wikipedia.page("Python")
# %%
# page = wikipedia.page("Animalss", auto_suggest=False)
# %%
# Fixes PageError by allowing redirects
page = wikipedia.page("Animalss", redirect=True)
print(page.title)

# Fixes DisambiguationError by selecting the first option
try:
    page = wikipedia.page("Python")
except DisambiguationError as e:
    page = wikipedia.page(e.options[0])
print(page.title)
# %%
def get_page(title: str) -> WikipediaPage:
    """
    Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option. If the title is not found, try to find a similar title.

    Args:
        title (str): The title of the Wikipedia page

    Returns:
        WikipediaPage: The Wikipedia page
    """
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
    except PageError as e:
        return wikipedia.page(title, auto_suggest=True, redirect=True)

# %%
def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    all_links = current_page.links
    content = current_page.content
    return [link for link in all_links if link in content]

# %%
wiki_page = get_page("Wikipedia")
links = get_permitted_links(wiki_page)
print(links)
# %%
print("Arabic Wikipedia" in wiki_page.links)
# %%
class WikiGame:
    def __init__(
        self,
        starting_page: str,
        goal_page: str,
    ):
        """
        Initialize the Wikipedia game object.

        Args:
            starting_page (str): The page the agent starts on.
            goal_page (str): The page the agent is trying to reach.
        """

        # Task state variables
        self.page_history: List[str] = [starting_page]
        self.starting_page: WikipediaPage = self.get_page(starting_page)
        self.goal_page: WikipediaPage = self.get_page(goal_page)
        self.current_page: WikipediaPage = self.starting_page

    # ========================= Helper Functions (given) =========================

    # Get page and page summary
    @staticmethod
    def get_page(title: str) -> WikipediaPage:
        """
        Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option. If the title is not found, try to find a similar title.

        Args:
            title (str): The title of the Wikipedia page

        Returns:
            WikipediaPage: The Wikipedia page
        """
        try:
            return wikipedia.page(title, auto_suggest=False, redirect=True)
        except DisambiguationError as e:
            return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
        except PageError as e:
            return wikipedia.page(title, auto_suggest=True, redirect=True)

    def get_page_summary(self, page: WikipediaPage | None = None) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (WikipediaPage): The Wikipedia page object.

        Returns:
            str: The summary of the Wikipedia page.
        """
        page = page if page else self.goal_page
        summary = page.content[:500]
        last_period_index = summary.rfind(".")
        return summary[: last_period_index + 1] if last_period_index != -1 else summary

    # Get and check permitted links
    def get_permitted_links(self, title: Optional[str] = None) -> list[str]:
        """
        Returns a list of permitted links (i.e. links in the main page content) for the current page.

        Args:
            title (Optional[str]): The title of the Wikipedia page. If None, uses the current page.

        Returns:
            list[str]: The permitted links.
        """
        if title:
            page = self.get_page(title)
            all_links = page.links
            content = page.content
            permitted_links = [link for link in all_links if link in content]
            if title in permitted_links:
                permitted_links.remove(title)
        else:
            all_links = self.current_page.links
            content = self.current_page.content
            permitted_links = [link for link in all_links if link in content]
            if self.current_page.title in permitted_links:
                permitted_links.remove(self.current_page.title)
        return permitted_links

    def is_permitted_link(self, link: str) -> bool:
        """
        Returns True if the link is in the permitted links for the current page, False otherwise.

        Args:
            link (str): The link to check.

        Returns:
            bool: True if the link is permitted, False otherwise
        """
        return link.lower() in (x.lower() for x in self.get_permitted_links())

    # ========================= Task-specific instructions (to implement) =========================

    @property
    def system_instruction(self) -> dict:
        """
        Generate the starting instructions for the game, formatted as a system prompt.

        Returns:
            dict: The starting instructions. The "role" is "system" for system messages.
        """
        # TODO
        return {
            "role": "system",
            "content": f"You are an expert wikipedia-racing AI agent that knows all of the Wikipedia pages. You know all of the links in every Wikipedia page. You will try to take the fewest link clicks to get from the Wikipedia page: {self.starting_page.title} to the Wikipedia page: {self.goal_page.title}. Try to move between these pages by following the fewest number of links."
        }

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        # TODO
        return {"role": "user",
                "content": f"Remember we want to get from the current page {self.current_page.title} to the goal page {self.goal_page.title} by clicking links on the page with the smallest number of steps possible.\nYou are currently on page {self.current_page.title} and the summary of this page is: {self.get_page_summary(self.current_page)}"
                }
        
    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        # TODO
        # concatenated_links = '\n'.join(self.get_permitted_links(self.current_page.title))
        return {
            "role": "user",
            "content": f"Remember we want to get from the current page {self.current_page.title} to the goal page {self.goal_page.title} by clicking links on the page with the smallest number of steps possible.\nYou should choose the next link to click on.\nWhat's your next step?"
            # \nThe links on the current page are:\n{concatenated_links}.
        }

    # ========================= Task State management (to implement) =========================

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        # TODO
        return self.current_page.title == self.goal_page.title

tests.run_wiki_game_tests(WikiGame)
# %%

class GetContentTool():
    name: str = "get_content"

    @staticmethod
    def execute(task: WikiGame | Any) -> str:
        """
        Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.

        Args:
            task (WikiGame | Any): The current task object.

        Returns:
            str: The content of the page with links wrapped
        """
        content = task.current_page.content
        permitted_links = get_permitted_links(task.current_page)
        for word in sorted(permitted_links, key=len, reverse=True):
            content = re.sub(
                r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
                r"\1<link>\2</link>\3",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        return content

    @property
    def description(self):
        """
        Provides the description of the GetContentTool.

        Returns:
            dict: The description of the GetContentTool for the API
        """
        # TODO
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the content of the current Wikipedia page. Call this function every time you get to a new page to find another link to click on. The links will be presented between '<link></link>' tags.",
                "parameters": {
                    "type": "object",
                    "properties" : {
                    },
                    "required": [],
                    "additionalProperties": False
                },
            }
        }

# %%
class MovePageTool():
    name: str = "move_page"

    @staticmethod
    def execute(new_page: str, task: Any) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Args:
            task (WikiGame): The current task object.
            new_page (str): The title of the new page to move to.

        Returns:
            str: A message indicating the result of the move
        """
        # TODO
        new_page_normalised = new_page.replace("_", " ")
        if not task.is_permitted_link(new_page_normalised):
            return f"Could not move page to {new_page}. This is not a valid link."
        task.current_page = task.get_page(new_page_normalised)
        task.page_history.append(task.current_page.title)
        return f"Moving page to {task.current_page.title}"

    @property
    def description(self):
        """
        Provides the description of the MovePageTool

        Returns:
            dict: The description of the MovePageTool for the API
        """
        # TODO
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": 'The title of the new page you want to move to. This should be formatted the way the title appears on wikipedia (e.g. to move to the wikipedia page for the United States of America, you should enter "United States"). Underscores are not necessary.',
                        }
                    },
                    "required": ["new_page"],
                },
            },
        }


get_content_tool_inst = GetContentTool()
move_page_tool_inst = MovePageTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst]

# %%
class WikiAgent(SimpleAgent):
    """
    Inherits from SimpleAgent and adds the ability to handle tool calls and refusals in the Wikipedia game context.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage:
            Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]:
            Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool:
            Run one loop of the Wikipedia agent (modified below)

    """

    def __init__(
        self,
        task: Any,
        tools: List[Any],
        model="gpt-4o-mini",
        chat_history: List[dict] = None,
        verbose: bool = True,
    ):
        super().__init__(model=model, tools=tools, task=task)

        self.chat_history = chat_history if chat_history else []
        self.full_chat_history = (
            chat_history if chat_history else []
        )  # All messages that have been sent in the chat history.
        self.verbose = verbose
        self.start()

    # ========================= Memory (to implement) =========================

    def update_history(
        self, message: str | ChatCompletionMessage | List[str | ChatCompletionMessage]
    ):
        """
        Update self.chat_history and self.full_chat_history with a message or list of messages.

        Args:
            message (str | List[str]): The message to add to the chat history
        """
        if not isinstance(message, list):
            message = [message]
        message = [apply_user_format(m) if isinstance(m, str) else m for m in message]
        self.chat_history += message
        self.full_chat_history += message

    def reset_history(self):
        """
        Empty self.chat_history of the agent.
        """
        self.chat_history = [self.task.system_instruction]

    # ========================= Observation parsing (to implement) =========================
    def handle_tool_calls(self, response: ChatCompletionMessage):
        """
        Handles tool_calls in the wikipedia game context:
            - Executes the tool calls using execute_tool_calls
            - Appends the original tool call & tool_responses to the chat_history
            - If the agent has moved to a new page, resets the chat_history
            - If not, get the next_step_message instruction from the task and append it to chat_history

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        self.update_history(response)

        tool_call_results = self.execute_tool_calls(response)
        for tool_call, result in zip(response.tool_calls, tool_call_results):
            tool_call_result_message = apply_tool_call_format(tool_call, content=result)
            self.update_history(tool_call_result_message)
        # Move to new page if necessary
        if any("Moving page" in tool_response for tool_response in tool_call_results):
            self.reset_history()
            print(
                f"""{("-" * 50)} \n\nMOVED TO PAGE \n\nPATH HISTORY (N={len(self.task.page_history)}): {" -> ".join(self.task.page_history)} \n\n{("-"*50)}"""
            )

            # Give starting instructions if moved to a new page
            self.start()

        # Otherwise ask the agent what the next step is

        else:
            next_step_message = self.task.next_step_instruction
            self.update_history(next_step_message)
            if self.verbose:
                print(f"""\nUSER: \n{next_step_message["content"]}""")

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        self.update_history(response)

    # ========================= Implementation logic (to implement) =========================
    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        self.reset_history()
        self.update_history(self.task.system_instruction)
        self.update_history(self.task.on_page_instruction)
        self.update_history(self.task.next_step_instruction)

    def run(self):
        """
        This is the main function that runs the agent in the wikipedia game for 1 loop. It:
            - Gets the current task instruction
            - Gets the response from the model
            - Handles the response in the cases:
                - tool calls (using handle_tool_calls)
                - refusals (using handle_refusal)
                - no tool calls (using update_history)
        """
        print(f"Running WikiAgent...")

        response = self.get_response(use_tool=True)

        if response.refusal:
            self.handle_refusal(response)
        
        if response.tool_calls:
            self.update_history(response)
            self.handle_tool_calls(response)
        else:
            self.update_history(apply_assistant_format(response))

        winner_winner_chicken_dinner = self.task.check_win()

        print(f"{winner_winner_chicken_dinner=}")
        # print(f"{self.task.current_page.title=}")

        
# %%
def agent_loop(agent, game, num_loops=10) -> bool:
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """

    for _ in range(num_loops):
        winner_winner_chicken_dinner = agent.task.check_win()
        if winner_winner_chicken_dinner:
            return True
        
        agent.run()

    return False
# %%
game_1 = WikiGame("Barack Obama", "India")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
# agent_loop(agent, game_1, 30)
# %%

game_1 = WikiGame("Vladimir Putin", "Anal Sex")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
# agent_loop(agent, game_1, 30)

# %%

class WikiGamePrompting(WikiGame):
    """
    Inherits from WikiGame and adds improved prompting.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)

    Methods:
        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (modified below)

        on_page_instruction -> dict: Generate instructions for the current page (modified below)

        next_step_instruction -> dict: Generate instructions for the next step (modified below)

        get_instructions(system: bool, on_page: bool, next_step: bool) -> str: Generate instruction messages based on the current game state (inherited)

        check_win() -> bool: Check if the game has been won (inherited)

    """

    @property
    def system_instruction(self):
        """
        Provide improved starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        # TODO
        return {
            "role": "system",
            "content": f"You are playing the Wikipedia game. "
            f"You are an expert wikipedia-racing AI agent that knows all of the Wikipedia pages. You know all of the links in every Wikipedia page. "
            f"You will try to take the fewest link clicks to get from the Wikipedia page: {self.starting_page.title} to the Wikipedia page: {self.goal_page.title}. Try to move between these pages by following the fewest number of links."
        }

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """

        return {
            "role": "user",
            "content": 
            "Generally, a good strategy is going from the source page to a more popular page with lots of links, and then going from it to the destination page. "
            "To do the second step, please write everything you know about the destination page, and then when you go to narrow pages, only go to a page that is about a thing the destination page is part of. "
            "Do not go in loops: if you have visited a page, don't visit it again. Check your history for which links you have already clicked on. "
            "Before acting, please first write a plan on how you would go from the source page to the destination path. "
            "When writing your plan, please consider many possible paths. "
            "After writing your plan, think about which path is best. "
            "You are also encouraged to think about which paths are bad and, if any are, replace them with new ones. "
            "When you land on a new page, first look at its content. "
            "Then, reason out loud. "
            "Only then click on a link. "
            "IMPORTANT NOTE: The links are only the things between <link>...</link> tags. "
            "If you see something that is not between <link>...</link> tags, you cannot click it, because it is not a link. "
            # "You should click links often, because the number of messages you can send is limited. "
            f"You started at page {self.starting_page.title}. "
            f"You are currently on page {self.current_page.title}. "
            f"Your goal is to go to page {self.goal_page.title}. "
            + ( f"The links you have clicked so far were the ones on this path: " 
                    + " -> ".join(page.title() for page in self.page_history) + ". "
                if len(self.page_history) >= 2 else "You have not clicked any links yet." )
            + f"The summary of the current page, titled {self.current_page.title} is: {self.current_page.summary}"
        }

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        # TODO
        return {
            "role": "user",
            "content": f"You should now choose the next link to click. "
            f"IMPORTANT NOTE: The links are only the things between <link>...</link> tags. "
            f"If you see something that is not between <link>...</link> tags, you cannot click it, because it is not a link. "
            f"Reminder that the goal is to go to the page {self.goal_page.title}"
        }

#Improved WikiGame and WikiAgent
game = WikiGamePrompting("Linux", "Dana Carvey")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
# successfully_got_to_destination_page = agent_loop(agent, game, 30)
# print(f"{successfully_got_to_destination_page=}")

# %%

class WikiGamePrompting(WikiGame):
    """
    Inherits from WikiGame and adds improved prompting.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)

    Methods:
        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (modified below)

        on_page_instruction -> dict: Generate instructions for the current page (modified below)

        next_step_instruction -> dict: Generate instructions for the next step (modified below)

        check_win() -> bool: Check if the game has been won (inherited)

    """

    @property
    def system_instruction(self):
        """
        Provide improved starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        return {
            "role": "system",
            "content": f"You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}.",
        }

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {
            "role": "user",
            "content": f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n\nThe path you have taken so far is {" -> ".join(self.page_history)}.
            """,
        }

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """

        return {
            "role": "user",
            "content": f"What's your next step to get to {self.goal_page.title}?",
        }
    
#Improved WikiGame and WikiAgent
game = WikiGamePrompting("Linux", "Dana Carvey")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
# agent_loop(agent, game, 30)

# %%

for message in agent.full_chat_history:
    if isinstance(message, dict):
        role = message["role"]
        content = message["content"]
    else:
        role = message.role
        content = message.content
    print("=" * 25 + role + "=" * 25)
    print(content)

# %%

# Original WikiGame and WikiAgent
game = WikiGame("Linux", "Dana Carvey")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
# agent_loop(agent, game, 30)

# %%

class WikiGameReAct(WikiGamePrompting):
    """
    Inherits from WikiGame and adds the ReAct framework.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)

    Methods:

        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (inherited)

        on_page_instruction -> dict: Generate instructions for the current page (inherited)

        next_step_instruction -> dict: Generate instructions for the next step (inherited)

        check_win() -> bool: Check if the game has been won (inherited)

    """

    def __init__(self, starting_page: str, goal_page: str, tools=None):
        super().__init__(starting_page, goal_page)
        self.tools = tools

    @property
    def system_instruction(self):
        """
        Provided a description of the tools in the system message. When generate is called with tools this is redundant, but when generate is called without tools, this is useful.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        tool_descriptions = "\n".join([tool.description["function"]["name"] + ":" + tool.description["function"]["description"] for tool in self.tools])
        return {
            "role": "system",
            "content": f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. You have access to {str(len(self.tools))} tools, which are:\n{tool_descriptions}. Please use the test_path tool.""",
        }

class WikiAgentReAct(WikiAgent):
    """
    Inherits from WikiAgent and adds the ReAct framework.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool: Run one loop of the Wikipedia agent (inherited)

        update_history(message : str | ChatCompletionMessage | List[str | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)

        reset_history(): Empty self.chat_history of the agent. (inherited)

        handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)

        handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)

        start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)

        run(): This function runs the agent in the wikipedia game context. (inherited)


    """

    def generate_reason(self) -> ChatCompletionMessage:
        # Get the model to reason about the current state of the game and add the response to the messages (you may not want to give it tools for this)
        self.chat_history.append(
            apply_user_format(
                "Think carefully about your current situation and what actions you want to take to get closer to"
                + self.task.goal_page.title
                + "."
            )
        )
        response = self.get_response(use_tool=False)
        return response

    def generate_action(self) -> ChatCompletionMessage:
        # Get the model to generate an action based on the reasoning and add the response to the messages
        self.chat_history.append(apply_user_format("What action do you want to take?"))
        response = self.get_response(use_tool=True)
        # self.update_history(response)
        return response

    def generate_reason_and_action(self):
        """
        Generate a reason, store this in history, then generate and return an action.
        """
        reason = self.generate_reason()
        self.update_history(apply_assistant_format(reason.content))
        print("\nModel response ('Reason'):", reason.content)

        action = self.generate_action()

        return action

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        response = self.generate_reason_and_action()

        if response.tool_calls:
            self.handle_tool_calls(response)
        elif response.refusal:
            self.handle_refusal(response)

def agent_loop_ReAct(game, agent, num_loops = 10):
    """
    Run the agent loop for a given number of loops with the ReAct framework.

    Args:
        agent (WikiReActAgent): The agent to run
        game (WikiGameReAct): The game to play
        num_loops (int): The number of loops to run
    """
    agent.start()
    for i in range(num_loops):
        if game.check_win():
            print("Success")
            return
        agent.run()

# %%

# WikiGame and WikiAgent with improved prompting
game = WikiGamePrompting("Drupe", "17th parallel north")
agent = WikiAgent(task=game, tools=wiki_game_tools)
# agent_loop(agent, game, 40)

# %%

# WikiGame and WikiAgent with ReAct
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
# agent_loop_ReAct(game, agent,40)

# WikiGame and WikiAgent with ReAct
game = WikiGameReAct("Anal Sex", "Sepember 11 Attacks", tools=wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
# agent_loop_ReAct(game, agent,40)

# %%

class TestPathTool():
    """
    Implements a tool that allows the agent to test paths from the current state of the game.

    Attributes:
        name (str): The name of the tool

    Methods:
        execute(task: Any, path: str) -> str: Test if a given path is valid.

        description -> dict: Provides the description of the TestPathTool tool for the API
    """

    name = "test_path"

    def execute(self, task: Any, path: str) -> str:
        """
        Test if a given path is valid.

        Args:
            path (str): A string representing a path, e.g., "Barack Obama -> Indon
            esia -> India"

        Returns:
            str: A message indicating whether the path is valid or where it fails.
        """

        incorrect_link_message = ""
        path_valid = True
        for source_page, destination_page in pairwise(path.split("->")):
            links_on_source_page = get_permitted_links(get_page(source_page))
            source_page = source_page.strip().lower()
            destination_page = destination_page.strip().lower()
            links_on_source_page = [ link.strip().lower()
                                     for link in links_on_source_page ]
            link_exists = destination_page in links_on_source_page
            if not link_exists:
                incorrect_link_message += f" There is no link from page {source_page} to page {destination_page}."
                path_valid = False

        if path_valid:
            return "The path is valid."
        else:
            return "The path is not valid." + incorrect_link_message

    @property
    def description(self) -> dict:

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Given a path for the Wikipedia game, the tolo tells whether it is a valid path. If it is not a valid path, the tool tells which links on this path don't exist.",
                "parameters": {
                    "type": "object",
                    "properties" : {
                        "path" : {
                            "type": "string",
                            "description": "The path formatted as Barack Obama -> Indonesia -> India."
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False
                },
            }
        }

test_path_tool_inst = TestPathTool()
wiki_game_tools = [test_path_tool_inst, get_content_tool_inst, move_page_tool_inst]
# wiki_game_tools = [TestPathTool_inst]

game = WikiGameReAct("Linux", "Dana Carvey", tools = wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
# agent_loop_ReAct(game,agent, 40)

# %%

class WikiAgentChatHistory(WikiAgentReAct):
    """
    Inherits from WikiAgentReAct and adds the ability to store and retrieve chat history.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)
        full_chat_history (List[dict]): Full history of interactions

    Methods:
        - get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)
        - execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)
        - update_history(message : str | ChatCompletionMessage | List[str | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)
        - reset_history(): Empty self.chat_history of the agent. (modified below)
        - handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)
        - handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)
        - start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)
        - run(): This function runs 1 loop of the agent in the wikipedia game. (inherited)
        - store_chat_history(): Store the current chat history in the full chat history.
        - retrieve_chat_history(): Retrieve the full chat history.
    """
    def reset_history(self):
        """
        Replace the output of the get_content tool responses with "Wikipedia Content was output here" when the agent moves to a new page.

        This function should only be called if the agent moved to a new page. It should:
            - Look through the messages in the chat history
            - Determine if a message is a get_content tool call response
            - Replace the output of the get_content tool response with "Wikipedia Content was output here"
        """

        for message in self.full_chat_history:
            if isinstance(message, dict) and message["role"] == "tool" and "name" in "message" and message["name"] == "get_content":
                message["content"] = "Wikipedia Content was output here"
                print(message)
        super().reset_history()

game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentChatHistory(game, model="gpt-4o-mini", tools = wiki_game_tools)
# agent_loop_ReAct(game, agent, 40)
# %%

class GetAccessiblePageSummaryTool():
    """
    Implements a tool that allows the agent to get the summary of a Wikipedia page (you should use the get_page_summary function from the agent class)
    """

    name = "get_accessible_page_summary"

    @staticmethod
    def execute(task: Any, page_title: str) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (str): The Wikipedia page title.
            task (Any): The current task object.

        Returns:
            str: The summary of the Wikipedia page.
        """
        page = task.get_page(page_title)
        if page in task.get_permitted_links():
            return task.get_page_summary(task.current_page)
        else:
            return "This page is not accessible from the current page."

    @property
    def description(self):
        """
        Provides the description of the get_page_summary tool

        Returns:
            dict: The description of the tool for the API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the summary of a wikipedia page you are considering moving to, to the last full stop within the first 500 characters. The page needs to be accessible via a link from the current page. Anything which corresponds to a link you can select will be wrapped in <link></link> tags.",
                "parameters": {
                    "type": "object",
                    "properties" : {
                        "page_title" : {
                            "type": "string",
                            "description": "A summary of the Wikipedia page you want to get a summary of."
                        }
                    },
                    "required": ["page_title"],
                    "additionalProperties": False
                },
            }
        }


get_accessible_page_summary_tool_inst = GetAccessiblePageSummaryTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_accessible_page_summary_tool_inst]

# %%
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_accessible_page_summary_tool_inst]
game = WikiGameReAct("William Pitt the Younger", "Central Vietnam", tools = wiki_game_tools)
agent = WikiAgentChatHistory(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game, agent, 30)
# %%

class GetAnyPageContent():
    """
    Implements a tool that allows the agent to get a summary of any Wikipedia page (with no links wrapped in link tags).
    """

    name = "get_any_page_content"

    @staticmethod
    def execute(task: Any, page_title: str | None = None) -> str:
        """
        Get the content of any wikipedia page

        Also provides current page content if no page_title is provided.

        Args:
            page_title (str): The title of the Wikipedia page

        Returns:
            str: The content of the page (not wrapped in link tags).
        """


        return ""

    @property
    def description(self):
        """
        Provides the description of the get_any_page_content tool

        Returns:
            dict: The description of the tool for the API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get a summary of any Wikipedia. You will be able to preview the summary of a page without clicking on the link, so you can reason whether going there makes sense. This will not tell you which links are on the next page, it will only give you a summary of the page.",
                "parameters": {
                    "type": "object",
                    "properties" : {
                        "page_title" : {
                            "type": "string",
                            "description": "The title of the Wikipedia page you want to get a summary of."
                        }
                    },
                    "required": ["page_title"],
                    "additionalProperties": False
                },
            }
        }

get_any_page_content_tool_inst = GetAnyPageContent()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_any_page_content_tool_inst]