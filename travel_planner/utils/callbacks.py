from langchain_core.callbacks import BaseCallbackHandler

class TokenUsageTracker(BaseCallbackHandler):
    """
    LangChain callback handler to track and log token usage and costs for LLM invocations.
    """
    def __init__(self, step_name: str):
        self.step_name = step_name
        
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            for generation in response.generations:
                for g in generation:
                    if hasattr(g, 'message') and hasattr(g.message, 'response_metadata'):
                        metadata = g.message.response_metadata
                        token_usage = metadata.get('token_usage')
                        model_name = metadata.get('model_name', 'llama-3.1-8b-instant')
                        
                        if token_usage:
                            prompt_tokens = token_usage.get('prompt_tokens', 0)
                            completion_tokens = token_usage.get('completion_tokens', 0)
                            total_tokens = token_usage.get('total_tokens', 0)
                            
                            # Standard Groq API pricing estimation (in USD):
                            # Llama 3.1 8b: $0.05 / 1M input, $0.08 / 1M output
                            # Llama 3.3 70b: $0.59 / 1M input, $0.79 / 1M output
                            if "70b" in model_name.lower():
                                input_rate = 0.59
                                output_rate = 0.79
                            else:
                                input_rate = 0.05
                                output_rate = 0.08
                                
                            cost_usd = ((prompt_tokens * input_rate) + (completion_tokens * output_rate)) / 1_000_000
                            cost_inr = cost_usd * 83.5 # Approx USD to INR exchange rate
                            
                            print(f"\n📊 [LLM COST & TOKENS] {self.step_name}:")
                            print(f"   🤖 Model Used: {model_name}")
                            print(f"   📥 Input: {prompt_tokens} tokens | 📤 Output: {completion_tokens} tokens")
                            print(f"   🔀 Total: {total_tokens} tokens")
                            print(f"   💸 Cost: ${cost_usd:.6f} USD (~₹{cost_inr:.4f} INR)")
        except Exception as e:
            # Silently catch callback logging exceptions to prevent pipeline crash
            print(f"⚠️ Error parsing token metrics: {e}")
