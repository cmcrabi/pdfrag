from typing import List, Dict, Optional, Protocol
import logging
from openai import OpenAI
from google import genai
#from google.ai import generativelanguage as glm
from PIL import Image
from app.config import settings
import base64
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(self, query: str, context: str, images: List[Dict]) -> str:
        """Generate a response using the provider's model."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self):
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    async def generate_response(self, query: str, context: str, images: List[Dict]) -> str:
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._prepare_prompt(query, context)}
            ]

            # Add image messages if available
            if images:
                image_messages = self._prepare_image_messages(images)
                messages.extend(image_messages)

            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise

    def _get_system_prompt(self) -> str:
        return """You are a technical documentation expert specializing in software development, 
        system architecture, and technical manuals. Your task is to analyze and present technical information 
        in a clear, structured, and professional manner.

        Guidelines:
        1. Maintain technical accuracy while making the content accessible
        2. Use appropriate technical terminology
        3. Structure the response in a logical flow
        4. Include relevant code snippets or technical details when appropriate
        5. Reference diagrams, images, or tables when they are provided
        6. Format the response in Markdown
        7. Use headings, bullet points, and code blocks appropriately
        8. If images are present, describe them in context and reference them naturally
        9. For each image, provide a detailed description of its content and relevance to the topic
        10. Use image references in the format: [Image: description of the image]"""

    def _prepare_prompt(self, query: str, context: str) -> str:
        return f"""
        You are a technical documentation expert specializing in software development, 
        system architecture, and technical manuals. Your task is to analyze and present technical information 
        in a clear, structured, and professional manner.

        Guidelines:
        1. Maintain technical accuracy while making the content accessible
        2. Use appropriate technical terminology
        3. Structure the response in a logical flow
        4. Include relevant code snippets or technical details when appropriate
        5. Reference diagrams, images, or tables when they are provided
        6. Format the response in Markdown
        7. Use headings, bullet points, and code blocks appropriately
        8. For each image, use the following format:
           [IMAGE:image_path]
           Example: [IMAGE:data/processed/1/images/page_38_img_0.png]

        Query: {query}

        Context from relevant documents:
        {context}

        Available Images:
        {self._get_image_paths()}

        Please provide a comprehensive response that:
        1. Directly addresses the query
        2. Uses the provided context accurately
        3. References any images or diagrams when relevant
        4. Is formatted in Markdown
        5. Maintains technical accuracy
        6. For each image reference, use the [IMAGE:path] format
        7. Ensure all image paths match exactly with the available images listed above
        8. Place image references immediately after the relevant text they illustrate
        """

    def _get_image_paths(self) -> str:
        """Get a formatted string of available image paths."""
        if not hasattr(self, '_image_paths'):
            return "No images available"
        
        return "\n".join([
            f"- {path}" for path in self._image_paths
        ])

    def _prepare_image_messages(self, images: List[Dict]) -> List[Dict]:
        image_messages = []
        for img in images:
            image_path = Path(img['path'])
            if image_path.exists():
                try:
                    with open(image_path, 'rb') as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        image_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Image from page {img.get('page_number', 'unknown')}:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        })
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {str(e)}")
        return image_messages

class GeminiProvider(LLMProvider):
    """Google Gemini implementation of LLM provider."""
    
    def __init__(self):
        try:
            # Initialize Gemini client with API key
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_id = "gemini-2.0-flash"
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

    async def generate_response(self, query: str, context: str, images: List[Dict]) -> str:
        try:
            # Store image paths for the prompt
            self._image_paths = [img['path'] for img in images]
            
            # Prepare the prompt
            prompt = self._prepare_prompt(query, context)
            
            # Prepare content parts
            contents = [prompt]
            
            # Add images if available
            pil_images = [Image.open(path) for path in self._image_paths if Path(path).exists()]
            contents.extend(pil_images)

            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents
            )

            return response.text

        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            raise

    def _get_system_prompt(self) -> str:
        return """You are a technical documentation expert specializing in software development, 
        system architecture, and technical manuals. Your task is to analyze and present technical information 
        in a clear, structured, and professional manner.

        Guidelines:
        1. Maintain technical accuracy while making the content accessible
        2. Use appropriate technical terminology
        3. Structure the response in a logical flow
        4. Include relevant code snippets or technical details when appropriate
        5. Reference diagrams, images, or tables when they are provided
        6. Format the response in Markdown
        7. Use headings, bullet points, and code blocks appropriately
        8. If images are present, describe them in context and reference them naturally
        9. For each image, provide a detailed description of its content and relevance to the topic
        10. Use image references in the format: [Image: description of the image]"""

    def _prepare_prompt(self, query: str, context: str) -> str:
        return f"""
        You are a technical documentation expert specializing in software development, 
        system architecture, and technical manuals. Your task is to analyze and present technical information 
        in a clear, structured, and professional manner.

        Guidelines:
        1. Maintain technical accuracy while making the content accessible
        2. Use appropriate technical terminology
        3. Structure the response in a logical flow
        4. Include relevant code snippets or technical details when appropriate
        5. Reference diagrams, images, or tables when they are provided
        6. Format the response in Markdown
        7. Use headings, bullet points, and code blocks appropriately
        8. For each image, use the following format:
           [IMAGE:image_path]
           Example: [IMAGE:data/processed/1/images/page_38_img_0.png]

        Query: {query}

        Context from relevant documents:
        {context}

        Available Images:
        {self._get_image_paths()}

        Please provide a comprehensive response that:
        1. Directly addresses the query
        2. Uses the provided context accurately
        3. References any images or diagrams when relevant
        4. Is formatted in Markdown
        5. Maintains technical accuracy
        6. For each image reference, use the [IMAGE:path] format
        7. Ensure all image paths match exactly with the available images listed above
        8. Place image references immediately after the relevant text they illustrate
        """

    def _get_image_paths(self) -> str:
        """Get a formatted string of available image paths."""
        if not hasattr(self, '_image_paths'):
            return "No images available"
        
        return "\n".join([
            f"- {path}" for path in self._image_paths
        ])

class LLMService:
    """Main LLM service that uses the appropriate provider."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = self._get_provider(provider)

    def _get_provider(self, provider: str) -> LLMProvider:
        if provider.lower() == "openai":
            return OpenAIProvider()
        elif provider.lower() == "gemini":
            return GeminiProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def generate_response(self, query: str, search_results: List[Dict], context_pages: int = 3) -> str:
        """
        Generate a comprehensive response based on search results.
        """
        try:
            # Prepare the context from search results
            context = self._prepare_context(search_results)
            
            # Prepare images from search results
            images = self._prepare_images(search_results)

            # Generate response using the selected provider
            return await self.provider.generate_response(query, context, images)

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise

    def _prepare_context(self, search_results: List[Dict]) -> str:
        """
        Prepare the context from search results in a format suitable for the LLM.
        """
        context_parts = []
        
        for group in search_results:
            # Add the main context (high similarity match)
            if group.get('context'):
                context_parts.append(f"\nMain Context (Page {group['context']['metadata']['page_number']}):")
                context_parts.append(group['context']['content'])
            
            # Add related pages
            if group.get('pages'):
                for page in group['pages']:
                    context_parts.append(f"\nRelated Page {page['metadata']['page_number']}:")
                    context_parts.append(page['content'])
        
        return "\n".join(context_parts)

    def _prepare_images(self, search_results: List[Dict]) -> List[Dict]:
        """
        Prepare images from search results.
        """
        images = []
        
        for group in search_results:
            # Process images from main context
            if group.get('context') and group['context'].get('images'):
                for img in group['context']['images']:
                    images.append({
                        "path": img['path'],
                        "page_number": group['context']['metadata']['page_number']
                    })
            
            # Process images from related pages
            if group.get('pages'):
                for page in group['pages']:
                    if page.get('images'):
                        for img in page['images']:
                            images.append({
                                "path": img['path'],
                                "page_number": page['metadata']['page_number']
                            })
        
        return images 