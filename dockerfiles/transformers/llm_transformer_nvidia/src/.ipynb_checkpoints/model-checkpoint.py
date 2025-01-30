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
        query = data["query"]
        context = data["context"]

        logger.info(f"🛠 Received request: {json.dumps(data, indent=2)}")

        # ✅ Ensure correct predictor URL
        predictor_url = f"http://{self.predictor_host}/v1/chat/completions"
        logger.info(f"📡 Sending request to LLM predictor at {predictor_url}")

        llm_payload = {
            "model": "meta/llama-2-7b-chat",
            "messages": [
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            "temperature": data.get("temperature", 0.5),  # 👈 Dynamic now!
            "top_p": data.get("top_p", 1),
            "max_tokens": int(data.get("max_tokens", 256)),  # 👈 Should reflect changes!
            "stream": False
        }

        logger.info(f"📩 Payload sent to LLM: {json.dumps(llm_payload, indent=2)}")

        llm_response = requests.post(predictor_url, json=llm_payload, verify=False)

        if llm_response.status_code == 200:
            result = llm_response.json()["choices"][0]["message"]["content"]
            logger.info(f"✅ LLM Response: {result}")
            return {"predictions": [result]}
        else:
            error_message = f"❌ Error calling LLM predictor: {llm_response.status_code} - {llm_response.text}"
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
