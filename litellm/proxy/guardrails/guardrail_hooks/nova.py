"""
NOVA: The Prompt Pattern Matching
Authors: Thomas Roccia & Paolo Di Prodi
twitter: @fr0gger_
License: MIT License
Version: 1.0.0
Description: Command-line tool for running Nova rules against prompts for LiteLLM

"""
from typing import Literal, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth

# Nova dependencies
# Filter out the specific FutureWarning about clean_up_tokenization_spaces
import warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# Set transformers tokenization settings at the very beginning
# This must be before any imports to prevent FutureWarning
try:
    import transformers
    # Explicitly set the tokenization spaces parameter to prevent FutureWarning
    if hasattr(transformers, 'tokenization_utils_base'):
        transformers.tokenization_utils_base.CLEAN_UP_TOKENIZATION_SPACES = True
    if hasattr(transformers, 'PreTrainedTokenizerBase'):
        transformers.PreTrainedTokenizerBase.clean_up_tokenization_spaces = True
except ImportError:
    pass

import os
import sys
import argparse
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import colorama
from colorama import Fore, Style, Back

GUARDRAIL_NAME = "nova"

# Import Nova components
try:
    from nova.core.parser import NovaParser
    from nova.core.matcher import NovaMatcher
    from nova.core.scanner import NovaScanner
    from nova.utils.config import get_config
    from nova.evaluators.llm import (
        OpenAIEvaluator, 
        AnthropicEvaluator, 
        AzureOpenAIEvaluator, 
        OllamaEvaluator,
        GroqEvaluator,
        get_validated_evaluator
    )
except ImportError:
    verbose_proxy_logger.error("Error: Nova package not found in PYTHONPATH.")
    verbose_proxy_logger.error("Make sure Nova is installed or set your PYTHONPATH correctly.")

def load_rule_file(file_path: str) -> Optional[str]:
    """
    Load a Nova rule from a file.
    
    Args:
        file_path: Path to the rule file
        
    Returns:
        String containing the rule definition
        
    Raises:
        FileNotFoundError: If the rule file doesn't exist
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        verbose_proxy_logger.error(f"Error: Rule file not found: {file_path}")
        return None
    except Exception as e:
        verbose_proxy_logger.error(f"Error loading rule file: {e}")
        return None


def load_prompts_file(file_path: str) -> List[str]:
    """
    Load a list of prompts from a file.
    
    Args:
        file_path: Path to the prompts file
        
    Returns:
        List of prompts
        
    Raises:
        FileNotFoundError: If the prompts file doesn't exist
    """
    try:
        with open(file_path, 'r') as f:
            # Remove empty lines and strip whitespace
            prompts = [line.strip() for line in f.readlines()]
            prompts = [p for p in prompts if p and not p.startswith('#')]
            return prompts
    except FileNotFoundError:
        verbose_proxy_logger.error(f"Error: Prompts file not found: {file_path}")
        return []
    except Exception as e:
        verbose_proxy_logger.error(f"Error loading prompts file: {e}")
        return []


def extract_rules(content: str) -> List[str]:
    """
    Extract individual rule blocks from a file containing multiple rules.
    
    Args:
        content: String containing multiple rule definitions
        
    Returns:
        List of strings, each containing a single rule
    """
    # Pattern to find rule declarations
    rule_start_pattern = r'rule\s+\w+\s*{?'
    rule_starts = [m.start() for m in re.finditer(rule_start_pattern, content)]
    
    if not rule_starts:
        return []
    
    # Extract each rule block
    rule_blocks = []
    
    for i in range(len(rule_starts)):
        start = rule_starts[i]
        
        # End is either the start of the next rule or the end of the content
        end = rule_starts[i+1] if i < len(rule_starts) - 1 else len(content)
        
        # Extract the rule text
        rule_text = content[start:end].strip()
        
        # Ensure the rule has a closing brace
        if not _has_balanced_braces(rule_text):
            # Try to find where the rule should end
            possible_end = _find_rule_end(rule_text)
            if possible_end > 0:
                rule_text = rule_text[:possible_end].strip()
        
        rule_blocks.append(rule_text)
    
    return rule_blocks


def _has_balanced_braces(text: str) -> bool:
    """Check if braces are balanced in a text."""
    count = 0
    
    for char in text:
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
            
        # Negative count means too many closing braces
        if count < 0:
            return False
    
    # All braces should be closed
    return count == 0


def _find_rule_end(text: str) -> int:
    """Find the most likely end position of a rule."""
    # Count opening braces
    count = 0
    for i, char in enumerate(text):
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
            
            # When count reaches 0, we've found the end of the rule
            if count == 0:
                return i + 1
    
    return -1


def check_if_rule_needs_llm(rule) -> bool:
    """
    Check if a rule requires LLM evaluation based on its patterns and condition.
    
    Args:
        rule: The parsed Nova rule
        
    Returns:
        Boolean indicating whether LLM evaluation is needed
    """
    # Check if the rule has LLM patterns
    if hasattr(rule, 'llms') and rule.llms:
        return True
        
    # Check if the condition references LLM evaluation
    if hasattr(rule, 'condition') and rule.condition and 'llm.' in rule.condition.lower():
        return True
        
    return False


def check_if_rules_need_llm(rules) -> bool:
    """
    Check if any rule in a list requires LLM evaluation.
    
    Args:
        rules: List of parsed Nova rules
        
    Returns:
        Boolean indicating whether any rule needs LLM evaluation
    """
    for rule in rules:
        if check_if_rule_needs_llm(rule):
            return True
    
    return False


def process_prompt(rule_text: str, prompt: str, verbose: bool = False, 
                   llm_type: str = 'openai', model: Optional[str] = None,
                   llm_evaluator: Optional[Any] = None) -> Dict[str, Any]:
    """
    Process a prompt against a rule.
    
    Args:
        rule_text: Nova rule definition
        prompt: Prompt to check
        verbose: Whether to enable verbose output
        llm_type: Type of LLM evaluator to use ('openai', 'anthropic', 'azure', or 'ollama')
        model: Optional model name to use
        llm_evaluator: Optional pre-existing LLM evaluator to reuse
        
    Returns:
        Dictionary containing match results or None if processing failed
    """
    # Parse the rule
    parser = NovaParser()
    
    try:
        rule = parser.parse(rule_text)
    except Exception as e:
        verbose_proxy_logger.error(f"Error parsing rule: {e}")
        pass
        return None
    
    # Check if this rule needs LLM evaluation
    needs_llm = check_if_rule_needs_llm(rule)
    
    # Use provided evaluator or create one if needed
    if needs_llm and not llm_evaluator:
        llm_evaluator = get_validated_evaluator(llm_type, model, verbose)
        if llm_evaluator is None:
            verbose_proxy_logger.error(f"Error: Failed to create LLM evaluator but rule requires it.")
            pass
            return None
    elif not needs_llm:
        if verbose:
            print(f"{Fore.GREEN}Rule '{rule.name}' only uses keyword/semantic matching. Skipping LLM evaluator creation.")
    
    # Match the prompt against the rule
    matcher = NovaMatcher(rule, llm_evaluator=llm_evaluator)
    
    # Handle None prompts safely
    if prompt is None:
        return {
            "matched": False,
            "rule_name": rule.name,
            "meta": rule.meta,
            "matching_keywords": {},
            "matching_semantics": {},
            "matching_llm": {},
            "debug": {
                "condition": rule.condition,
                "condition_result": False,
                "all_keyword_matches": {},
                "all_semantic_matches": {},
                "all_llm_matches": {},
                "llm_info": {
                    "type": llm_type if needs_llm else "none",
                    "model": getattr(llm_evaluator, 'model', None) if needs_llm else None
                }
            }
        }
    
    # Process the prompt
    result = matcher.check_prompt(prompt)
    
    # Add LLM info to debug info
    if "debug" not in result:
        result["debug"] = {}
    
    result["debug"]["llm_info"] = {
        "type": llm_type if needs_llm else "none",
        "model": getattr(llm_evaluator, 'model', None) if needs_llm and llm_evaluator else None
    }
    
    return result


class NovaGuardrail(CustomGuardrail):
    def __init__(
        self,
        rule,
        single=False,

        **kwargs,
    ):
        """
        rule: Path to the Nova rule file
        single: Check only the first rule
        """
        # store kwargs as optional_params
        self.optional_params = kwargs
        # rule path t a file containing 
        super().__init__(**kwargs)

        # Load the rule file
        file_content = load_rule_file(rule)

        # Check if the file might contain multiple rules
        if not single and 'rule ' in file_content.lower() and file_content.count('rule ') > 1:
            # Extract all rules from the file
            rule_blocks = extract_rules(file_content)
            
            if not rule_blocks:
                verbose_proxy_logger.debug(f"{Fore.RED}No valid rules found in {rule}")
                #TODO: shall EXIT
                self.fail = True
                
            verbose_proxy_logger.debug(f"\n{Fore.CYAN}Found {Fore.WHITE}{len(rule_blocks)}{Fore.CYAN} rules in {Fore.WHITE}{rule}")

            # Parse all rules first
            parser = NovaParser()
            parsed_rules = []
            for rule_idx, rule_text in enumerate(rule_blocks):
                try:
                    rule = parser.parse(rule_text)
                    parsed_rules.append(rule)
                except Exception as e:
                    verbose_proxy_logger.debug(f"{Fore.RED}Error parsing rule #{rule_idx+1}: {e}")
            
            if not parsed_rules:
                verbose_proxy_logger.debug(f"{Fore.RED}Failed to parse any rules. Exiting.")
                #TODO: shall EXIT
                self.fail = True


    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        # In this guardrail, if a user inputs `litellm` we will mask it and then send it to the LLM
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        _content = _content.replace("litellm", "********")
                        message["content"] = _content

        verbose_proxy_logger.debug(
            "async_pre_call_hook: Message after masking %s", _messages
        )

        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """

        # this works the same as async_pre_call_hook, but just runs in parallel as the LLM API Call
        # In this guardrail, if a user inputs `litellm` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        raise ValueError("Guardrail failed words - `litellm` detected")

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response

        If a response contains the word "coffee" -> we will raise an exception
        """
        verbose_proxy_logger.debug("async_pre_call_hook response: %s", response)
        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    verbose_proxy_logger.debug("async_pre_call_hook choice: %s", choice)
                    if (
                        choice.message.content
                        and isinstance(choice.message.content, str)
                        and "coffee" in choice.message.content
                    ):
                        raise ValueError("Guardrail failed Coffee Detected")
