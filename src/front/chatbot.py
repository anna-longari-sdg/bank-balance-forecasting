from openai import AzureOpenAI
import os

class BasicChatbot:
    def __init__(self, config: dict):
        print("Initializing BasicChatbot")
        self.config = config.get("chatbot", {}).copy()
        self.client = self._initialize_client()
        self.messages = [] # print: [{"role": "system", "content": "You are a helpful assistant."}]

    def _initialize_client(self):
        """
        Initializes the chatbot client based on the configuration.
        
        Returns
        -------
        client: AzureOpenAI
            An instance of the AzureOpenAI client configured with the settings from the config.
        """
        if self.config.get("type", '') == "openai":
            client = AzureOpenAI()
            self.config['deployment'] = os.getenv("OPENAI_DEPLOYMENT")
        else:
            raise ValueError(f"Unsupported embedder type: {self.config.get('type')}")
    
        return client
       
    def get_stream(self,):
        """
        Generates a response stream from the chatbot client.
        
        Returns
        -------
        stream: generator
            A generator that yields responses from the chatbot client.
        """
        stream = self.client.chat.completions.create(
            model=self.config.get("deployment"),
            messages=self.messages,
            stream=True,
        )
        return stream