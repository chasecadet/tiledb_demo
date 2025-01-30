import json
import logging
import argparse
import requests
from kserve import Model, ModelServer, model_server

logger = logging.getLogger(__name__)

class Transformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str,
                 use_ssl: bool, vectorstore_name: str = "vectorstore"):
        super().__init__(name)
        self.name = name
        self.predictor_host = predictor_host  # ✅ KServe passes this correctly
        self.protocol = protocol
        self.use_ssl = use_ssl
        self.ready = True

        # Transformer-specific arguments
        self.vectorstore_name = vectorstore_name
        self.vectorstore_url = self._build_vectorstore_url()

    def _get_namespace(self):
        return open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r").read().strip()

    def _build_vectorstore_url(self):
        domain_name = "svc.cluster.local"
        namespace = self._get_namespace()
        svc = f"{self.vectorstore_name}-predictor.{namespace}.{domain_name}"
        return f"http://{svc}/v1/models/{self.vectorstore_name}:predict"

    def preprocess(self, request: dict, headers: dict) -> dict:
        """
        Preprocess request: If no context, retrieve documents.
        """
        data = request["instances"][0]
        query = data["input"]
        logger.info(f"Received question: {query}")
        
        num_docs = data.get("num_docs", 4)
        context = data.get("context", None)

        if not context:
            payload = {"instances": [{"input": query, "num_docs": num_docs}]}
            logger.info(f"Retrieving relevant documents from: {self.vectorstore_url}")

            response = requests.post(self.vectorstore_url, json=payload, verify=False)
            response_data = response.json()

            if response.status_code == 200 and "predictions" in response_data:
                context = "\n".join(response_data["predictions"])
            else:
                context = "No relevant documents found."

        logger.info(f"Retrieved Context:\n{context}")

        # ✅ Return structured response for predict() to process
        return {"instances": [{"query": query, "context": context}]}

    def predict(self, request: dict, headers: dict) -> dict:
        """
        Call LLM with retrieved context and return the response.
        """
        data = request["instances"][0]

        # ✅ Dynamically retrieve user inputs from request
        query = data.get("input", "")  # Ensure query isn't missing
        context = data.get("context", "")  # Default to empty context if not provided
        system_message = data.get("system", "You are an AI assistant.")  # Dynamic system message
        instruction = data.get("instruction", "Use the following context to answer the question.")

        # ✅ Ensure correct predictor URL
        predictor_url = f"http://{self.predictor_host}/v1/chat/completions"
        logger.info(f"Sending request to LLM predictor at {predictor_url}")

        # ✅ Allow dynamic values for LLM parameters
        llm_payload = {
            "model": data.get("model", "meta/llama-2-7b-chat"),  # Use provided model, default to Llama-2-7B
            "messages": [
                {"role": "system", "content": system_message},  # Custom system message
                {"role": "user", "content": f"{instruction}\n\nContext: {context}\n\nQuestion: {query}"}
            ],
            "temperature": float(data.get("temperature", 0.5)),  # Convert to float for safety
            "top_p": float(data.get("top_p", 1)),  # Convert to float
            "max_tokens": int(data.get("max_tokens", 256)),  # Convert to int
            "stream": data.get("stream", False)  # Boolean support
        }

        # 🔥 Log Payload to Debug
        logger.info(f"Payload to LLM: {json.dumps(llm_payload, indent=2)}")

        # ✅ Send request to LLM predictor
        llm_response = requests.post(predictor_url, json=llm_payload, verify=False)

        if llm_response.status_code == 200:
            result = llm_response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
            logger.info(f"LLM Response: {result}")
            return {"predictions": [result]}
        else:
            error_message = f"Error calling LLM predictor: {llm_response.status_code} - {llm_response.text}"
            logger.error(error_message)
            return {"predictions": [error_message]}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument("--predictor_host", help="The URL for the model predict function", required=True)
    parser.add_argument("--protocol", help="The protocol for the predictor", default="v1")
    parser.add_argument("--model_name", help="The name that the model is served under.")
    parser.add_argument("--use_ssl", help="Use SSL for connecting to the predictor", action='store_true')
    parser.add_argument("--vectorstore_name", default="vectorstore", required=False, help="The name of the Vector Store Inference Service")
    
    args, _ = parser.parse_known_args()
    logger.info(args)

    model = Transformer(args.model_name, args.predictor_host, args.protocol, args.use_ssl, args.vectorstore_name)
    ModelServer().start([model])
