# Business Logic Adapter Examples

This directory contains practical examples of how to create custom business logic adapters for different use cases.

## Quick Start Template

```python
from adapters.business_logic_adapter import BusinessLogicAdapter

class MyCustomAdapter(BusinessLogicAdapter):
    """
    Template for creating custom business logic adapters.
    This adapter provides context - your application provides the logic.
    """
    
    def __init__(self, custom_settings=None):
        super().__init__(custom_settings)
        
        # Initialize your domain-specific settings
        self.domain = custom_settings.get('domain', 'general')
        self.safety_level = custom_settings.get('safety_level', 'standard')
        
    def get_enhanced_instructions(self, original_instructions):
        """Provide domain-specific context to the AI"""
        
        context = f"""
        {original_instructions}
        
        DOMAIN CONTEXT:
        - You are assisting with {self.domain} related queries
        - Safety level: {self.safety_level}
        
        IMPORTANT BOUNDARIES:
        - Stay within your knowledge domain
        - If uncertain, say "I don't have that specific information"
        - Never provide advice outside your role
        - Always prioritize user safety
        
        RESPONSE STYLE:
        - Be helpful and professional
        - Acknowledge limitations honestly
        - Provide accurate information within your domain
        """
        
        return context
```

## Real-World Examples

### Customer Service Adapter

```python
class CustomerServiceAdapter(BusinessLogicAdapter):
    """Context provider for customer service applications"""
    
    def get_enhanced_instructions(self, original_instructions):
        return f"""
        {original_instructions}
        
        CUSTOMER SERVICE CONTEXT:
        - You are a helpful customer service assistant
        - Gather information to help resolve customer issues
        - Be empathetic and professional
        - Escalate complex issues appropriately
        
        BOUNDARIES:
        - Do NOT promise refunds or compensation
        - Do NOT make policy decisions
        - Do NOT access customer account details
        - DO gather issue details and contact information
        - DO provide general product information
        - DO schedule callbacks or escalations
        
        ESCALATION TRIGGERS:
        - Billing disputes
        - Technical issues beyond basic troubleshooting
        - Requests for management
        - Legal or compliance matters
        """
```

### Healthcare Information Assistant

```python
class HealthcareInfoAdapter(BusinessLogicAdapter):
    """Context provider for healthcare information applications"""
    
    def get_enhanced_instructions(self, original_instructions):
        return f"""
        {original_instructions}
        
        HEALTHCARE INFORMATION CONTEXT:
        - You provide general health information only
        - Help users understand medical terminology
        - Assist with appointment scheduling and information gathering
        
        CRITICAL BOUNDARIES:
        - NEVER provide medical diagnosis
        - NEVER recommend specific treatments
        - NEVER interpret test results
        - NEVER provide emergency medical advice
        - DO provide general health education
        - DO help with appointment scheduling
        - DO explain medical procedures in simple terms
        
        EMERGENCY PROTOCOL:
        - For emergencies: "Please call emergency services immediately"
        - For urgent concerns: "Please contact your healthcare provider"
        - Always err on the side of caution
        """
```

### E-commerce Assistant

```python
class EcommerceAdapter(BusinessLogicAdapter):
    """Context provider for e-commerce applications"""
    
    def get_enhanced_instructions(self, original_instructions):
        return f"""
        {original_instructions}
        
        E-COMMERCE CONTEXT:
        - Help customers find products and information
        - Assist with order inquiries and basic support
        - Provide product information and comparisons
        
        CAPABILITIES:
        - Product search and recommendations
        - Order status inquiries (with order number)
        - General shipping and return policy information
        - Size guides and product specifications
        
        LIMITATIONS:
        - Cannot process payments or modify orders
        - Cannot access customer account details
        - Cannot make policy exceptions
        - Cannot handle returns without proper authorization
        
        ESCALATION:
        - Payment issues → Customer service team
        - Order modifications → Customer service team
        - Defective products → Quality assurance team
        """
```

## Integration Example

```python
# In your application
from universal_ai_sdk import UniversalAIClient, AgentConfig

# Initialize your custom adapter
custom_settings = {
    'domain': 'customer_service',
    'safety_level': 'high',
    'escalation_email': 'support@yourcompany.com'
}

# Create agent with your adapter
config = AgentConfig(
    instructions="You are a customer service assistant",
    capabilities=["text", "voice"],
    business_logic_adapter="custom_customer_service",
    custom_settings=custom_settings
)

client = UniversalAIClient()
agent = client.create_agent(config)

# Your application handles the user interaction
user_message = "I'm having trouble with my order #12345"
response = agent.send_message(user_message)

# Your application processes the response
if "escalation_needed" in response.metadata:
    # Route to human agent
    route_to_human_agent(user_message, response)
else:
    # Continue with AI assistance
    send_response_to_user(response.content)
```

## Testing Your Adapter

```python
# Test anti-hallucination protection
def test_adapter_safety():
    adapter = MyCustomAdapter()
    
    # Test response validation
    safe_response = "I can help you with that information."
    risky_response = "I believe you probably have this medical condition."
    
    assert adapter.is_safe_response(safe_response) == True
    assert adapter.is_safe_response(risky_response) == False
    
    print("✅ Anti-hallucination protection working")

# Test context generation
def test_context_generation():
    adapter = MyCustomAdapter({'domain': 'test'})
    instructions = adapter.get_enhanced_instructions("Base instructions")
    
    assert "DOMAIN CONTEXT" in instructions
    assert "BOUNDARIES" in instructions
    
    print("✅ Context generation working")
```

## Next Steps

1. **Copy the template** and customize it for your domain
2. **Define clear boundaries** for what the AI should and shouldn't do
3. **Test thoroughly** especially the anti-hallucination features
4. **Monitor in production** and refine based on real usage
5. **Document your adapter** for your team

Remember: Adapters provide context, your application provides the logic!
